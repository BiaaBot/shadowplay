// =====================================================
// SERVER
// =====================================================

const dns = require("dns");
dns.setDefaultResultOrder("ipv4first");

require("dotenv").config();

const express = require("express");
const OpenAI = require("openai");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "20mb" }));
app.use(express.static(__dirname));


// =====================================================
// OPENAI
// =====================================================

//Creates an OpenAI client using the API key stored in the env. file
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

//Stores the name of the OpenAI model that we're going to be using
const OPENAI_MODEL = "gpt-4o-mini";


// =====================================================
// MEMORY CACHES
// =====================================================

//Creates a Map that stores completed image generation results
const imageCache = new Map();

//Stores generation promises that are currently running, preventing two simultaneous requests from generating the same asset twice
const pendingGenerations = new Map();


// =====================================================
// HELPERS
// =====================================================

//Converts arbitary text into a consistent lowercase identifier that can be used safely as an asset key or cache key.
function normalizeKey(s) {
  return String(s || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9\-:]/g, "");
}

//Retrieves a required environment variable and throws an explicit error when the variable has not been configured.
function requireEnv(name) {
  const v = process.env[name];

  if (!v) {
    throw new Error(`Missing env var: ${name}`);
  }

  return v;
}

//Returns the configured ComfyUI base URL after removing any trailing slash so API paths can be appended consistently.
function comfyBase() {
  return requireEnv("COMFYUI_URL").replace(/\/+$/, "");
}

//Loads a ComfyUI workflow from a JSON file.
function loadWorkflowJson(workflowPath) {
  const full = path.isAbsolute(workflowPath)
    ? workflowPath
    : path.join(__dirname, workflowPath);

  const raw = fs.readFileSync(full, "utf-8");

  return JSON.parse(raw);
}

//Defines an asynchronous helper for making HTTP requests with a timeout.
async function fetchWithTimeout(
  url,
  options = {},
  timeoutMs = 60000
) {
  const ac = new AbortController();

  const id = setTimeout(() => {
    ac.abort();
  }, timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: ac.signal,
    });
  } finally {
    clearTimeout(id);
  }
}

// =====================================================
// COMFY HELPER FUNCTIONS
// =====================================================

//Sends a completed workflow to ComfyUI's prompt queue and returns the prompt ID needed to track the workflow's execution.
async function comfyQueuePrompt(
  workflow,
  client_id = "triztese-client"
) {
  const url = `${comfyBase()}/prompt`;

  const resp = await fetchWithTimeout(
    url,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        prompt: workflow,
        client_id,
      }),
    },
    60000
  );

  const text = await resp.text();

  if (!resp.ok) {
    throw new Error(
      `ComfyUI /prompt failed: ${resp.status} ${text}`
    );
  }

  const data = JSON.parse(text);

  if (!data.prompt_id) {
    throw new Error(
      "ComfyUI did not return prompt_id"
    );
  }

  return data.prompt_id;
}

//Polls ComfyUI history for completed generation jobs and updates their status.
async function comfyPollHistory(
  promptId,
  timeoutMs = 900000,
  pollMs = 1500
) {
  const url = `${comfyBase()}/history/${promptId}`;

  const start = Date.now();

  while (
    Date.now() - start <
    timeoutMs
  ) {
    const resp =
      await fetchWithTimeout(
        url,
        { method: "GET" },
        30000
      );

    const text = await resp.text();

    if (!resp.ok) {
      throw new Error(
        `ComfyUI /history failed: ${resp.status} ${text}`
      );
    }

    const data = JSON.parse(text);

    const entry = data?.[promptId];

    if (entry?.outputs) {
      return entry;
    }

    await new Promise((r) =>
      setTimeout(r, pollMs)
    );
  }

  throw new Error(
    "Timed out waiting for ComfyUI workflow"
  );
}

//Converts raw SVG code into a browser-readable data URL.
function svgTextToDataUrl(svgText) {
  const encoded = Buffer
    .from(svgText, "utf8")
    .toString("base64");

  return `data:image/svg+xml;base64,${encoded}`;
}

//Function responsible for locating and preparing the SVG output folder.
function getSvgOutputDirectory() {
  const outputDirectory =
    path.join(
      __dirname,
      "generated-svg"
    );

  if (
    !fs.existsSync(
      outputDirectory
    )
  ) {
    fs.mkdirSync(
      outputDirectory,
      {
        recursive: true
      }
    );
  }

  return outputDirectory;
}

//Searches a directory and its subdirectories for SVG files with a specified filename prefix and returns the most recently modified match.
function findSvgByPrefix(
  directory,
  filenamePrefix
) {
  if (
    !fs.existsSync(
      directory
    )
  ) {
    return null;
  }

  const entries =
    fs.readdirSync(
      directory,
      {
        withFileTypes: true
      }
    );

  const matches = [];

  for (const entry of entries) {
    const fullPath =
      path.join(
        directory,
        entry.name
      );

    if (entry.isDirectory()) {
      const nested =
        findSvgByPrefix(
          fullPath,
          filenamePrefix
        );

      if (nested) {
        matches.push(
          nested
        );
      }

      continue;
    }

    if (
      entry.isFile() &&
      entry.name
        .toLowerCase()
        .endsWith(".svg") &&
      entry.name.startsWith(
        filenamePrefix
      )
    ) {
      const stats =
        fs.statSync(
          fullPath
        );

      matches.push({
        fullPath,
        modifiedTime:
          stats.mtimeMs
      });
    }
  }

  if (
    matches.length === 0
  ) {
    return null;
  }

  matches.sort(
    (a, b) =>
      b.modifiedTime -
      a.modifiedTime
  );

  return matches[0];
}

//Polls the output directory until an SVG with the expected filename prefix appears or the file-wait timeout is reached.
async function waitForSvgFile(
  directory,
  filenamePrefix,
  timeoutMs = 15000
) {
  const startedAt =
    Date.now();

  while (
    Date.now() -
    startedAt <
    timeoutMs
  ) {
    const result =
      findSvgByPrefix(
        directory,
        filenamePrefix
      );

    if (result) {
      return result;
    }

    await new Promise(
      (resolve) =>
        setTimeout(
          resolve,
          250
        )
    );
  }

  throw new Error(
    `ComfyUI completed, but no SVG beginning with ` +
    `"${filenamePrefix}" was found in ${directory}.`
  );
}

// =====================================================
// PROMPT WORKFLOW
// =====================================================

// Loads and configures a ComfyUI workflow, submits it for generation, waits for completion, locates the resulting SVG file, and returns its text and browser-ready data URL.
async function runPromptWorkflow({
  workflowPath,
  positivePrompt,
  negativePrompt,
  seed
}) {
  const workflow =
    loadWorkflowJson(
      workflowPath
    );

  const positiveNodeId =
    requireEnv(
      "PROMPT_NODE_ID"                  //Reads the node ID containing the positive text prompt.
    );

  const negativeNodeId =
    requireEnv(
      "NEGATIVE_PROMPT_NODE_ID"         //Reads the node ID containing the negative prompt.
    );

  const svgOutputNodeId =
    requireEnv(
      "SVG_OUTPUT_NODE_ID"              //Reads the ID of the ComfyUI node responsible for saving SVG outputs.
    );

  const positiveNode =
    workflow[
    positiveNodeId
    ];

  const negativeNode =
    workflow[
    negativeNodeId
    ];

  const samplerNodeId =
    requireEnv(
      "SAMPLER_NODE_ID"                //Reads the node ID of the sampler (whose seed controls generation variation).
    );

  const svgOutputNode =
    workflow[
    svgOutputNodeId
    ];

  const samplerNode =
    workflow[
    samplerNodeId
    ];

  if (
    !positiveNode?.inputs
  ) {
    throw new Error(
      `Prompt node ${positiveNodeId} was not found in the workflow.`
    );
  }

  if (
    !negativeNode?.inputs
  ) {
    throw new Error(
      `Negative prompt node ${negativeNodeId} was not found in the workflow.`
    );
  }

  if (
    !svgOutputNode?.inputs
  ) {
    throw new Error(
      `SVG output node ${svgOutputNodeId} was not found in the workflow.`
    );
  }

  if (
    !samplerNode?.inputs
  ) {
    throw new Error(
      `Sampler node ${samplerNodeId} was not found in the workflow.`
    );
  }

  samplerNode.inputs.seed =
    Number.isSafeInteger(seed)
      ? seed
      : Math.floor(
        Math.random() *
        Number.MAX_SAFE_INTEGER
      );

  positiveNode.inputs.text =
    positivePrompt;

  negativeNode.inputs.text =
    negativePrompt;

  const svgOutputDirectory =                             //Retrieves the project’s SVG output directory and creates it if necessary.
    getSvgOutputDirectory();

  const filenamePrefix =                                //Creates a unique filename for evert generated asset based on the current time.
    `triztese_${Date.now()}_`;


  svgOutputNode.inputs.filename_prefix =
    filenamePrefix;

  svgOutputNode.inputs.append_timestamp =
    true;

  svgOutputNode.inputs.custom_output_path =
    svgOutputDirectory;

  console.log(
    "SVG output directory:",
    svgOutputDirectory
  );

  console.log(
    "SVG filename prefix:",
    filenamePrefix
  );

  const promptId =
    await comfyQueuePrompt(
      workflow
    );

  await comfyPollHistory(
    promptId
  );

  const svgFile =                                          //Waits until an SVG beggining with the unique prefix appears.
    await waitForSvgFile(
      svgOutputDirectory,
      filenamePrefix
    );

  console.log(
    "Generated SVG found:",
    svgFile.fullPath
  );

  const svgText =
    fs.readFileSync(
      svgFile.fullPath,
      "utf-8"
    );

  if (
    !svgText ||
    !svgText.includes("<svg")
  ) {
    throw new Error(
      `The generated file is not valid SVG: ${svgFile.fullPath}`
    );
  }

  return {
    svgText,

    dataUrl:
      svgTextToDataUrl(
        svgText
      ),

    mimeType:
      "image/svg+xml",

    sourcePath:
      svgFile.fullPath
  };
}

//Sanitizes a character prompt to ensure it describes a single isolated puppet figure.
async function sanitizeCharacterPrompt(promptText) {
  const schema = {
    type: "object",
    additionalProperties: false,
    required: ["ok", "prompt"],
    properties: {
      ok: { type: "boolean" },
      prompt: { type: "string" },
    },
  };

  const response = await openai.responses.create({
    model: OPENAI_MODEL,

    input: [
      {
        role: "system",

        content:
          "You are a strict prompt sanitizer for laser-cut shadow puppet character descriptions."
      },

      {
        role: "user",

        content:
          `Sanitize the following prompt so it describes ONLY a single isolated puppet figure suitable for laser-cut silhouette design. Remove any references to other people, groups, interaction, objects/props, scenery, or anything that implies interaction. If you can produce a safe single-figure prompt, return {ok: true, prompt: <sanitized prompt>}. If impossible to sanitize, return {ok: false, prompt: ""}.

Original prompt:\n${String(promptText || "").slice(0, 2000)}`,
      },
    ],

    text: {
      format: {
        type: "json_schema",
        name: "sanitize_character",
        strict: true,
        schema,
      },
    },
  });

  return JSON.parse(response.output_text || "{}");
}


// =====================================================
// STORYLINE SCHEMA
// =====================================================

//Creates a reusable JSON schema.
const STORYLINE_SCHEMA = {
  type: "object",

  additionalProperties: false,

  required: [
    "logline",
    "acts",
  ],

  properties: {
    logline: {
      type: "string",
    },

    acts: {
      type: "array",

      minItems: 5,
      maxItems: 5,

      items: {
        type: "object",

        additionalProperties: false,

        required: [
          "title",
          "summary",
          "beats",
        ],

        properties: {
          title: {
            type: "string",
          },

          summary: {
            type: "string",
          },

          beats: {
            type: "array",

            minItems: 3,
            maxItems: 8,

            items: {
              type: "string",
            },
          },
        },
      },
    },
  },
};

// =====================================================
// PUPPET PLAN SCHEMA
// =====================================================

//Defines the expected structure for the puppet asset plan returned by the planner.
const PUPPET_PLAN_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["characters", "acts"],

  properties: {
    characters: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "key", "prompt"],
        properties: {
          name: { type: "string" },
          key: { type: "string" },
          prompt: { type: "string" }
        }
      }
    },

    acts: {
      type: "array",
      minItems: 5,
      maxItems: 5,

      items: {
        type: "object",
        additionalProperties: false,
        required: [
          "actIndex",
          "actTitle",
          "actSummaryShort",
          "background",
          "characterKeys"
        ],

        properties: {
          actIndex: {
            type: "integer",
            minimum: 0,
            maximum: 4
          },

          actTitle: {
            type: "string"
          },

          actSummaryShort: {
            type: "string"
          },

          background: {
            type: "object",
            additionalProperties: false,
            required: ["key", "prompt"],
            properties: {
              key: { type: "string" },
              prompt: { type: "string" }
            }
          },

          characterKeys: {
            type: "array",
            items: {
              type: "string"
            }
          }
        }
      }
    }
  }
};


// =====================================================
// HEALTH
// =====================================================

//Checks whether the configured ComfyUI server is reachable and returns a short preview of its system statistics response.
app.get(
  "/api/comfy/health",
  async (_req, res) => {
    try {
      const url = `${comfyBase()}/system_stats`;

      const r =
        await fetchWithTimeout(
          url,
          { method: "GET" },
          25000
        );

      const body =
        await r.text();

      res.status(
        r.ok ? 200 : 502
      ).json({
        ok: r.ok,
        url,
        status: r.status,
        bodyPreview:
          body.slice(0, 200),
      });
    } catch (e) {
      res.status(502).json({
        ok: false,
        error: String(e),
      });
    }
  }
);


// =====================================================
// STORYLINE
// =====================================================

//Generates a structured five-act children's story outline from the user's prompt and validates the response against the storyline schema.
app.post(
  "/api/storyline",
  async (req, res) => {
    try {
      const { prompt } =
        req.body || {};

      if (!prompt) {
        return res
          .status(400)
          .send("Missing prompt");
      }

      const response =
        await openai.responses.create({
          model: OPENAI_MODEL,

          input: [
            {
              role: "system",

              content:
                "You are a children's fairytale writer.",
            },

            {
              role: "user",

              content:
                `${prompt}\n\n` +
                `Create EXACTLY 5 acts.\n` +
                `Each act must have:\n` +
                `- title\n` +
                `- summary\n` +
                `- 3 to 6 beats`,
            },
          ],

          text: {
            format: {
              type: "json_schema",

              name: "storyline",

              strict: true,

              schema:
                STORYLINE_SCHEMA,
            },
          },
        });

      res.json(
        JSON.parse(
          response.output_text
        )
      );
    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);

// =====================================================
// ACT REGENERATE
// =====================================================

//Regenerates one selected act while providing the complete storyline as context and returns the replacement act in the original schema format.
app.post(
  "/api/act/regenerate",
  async (req, res) => {
    try {
      const {
        prompt,
        storyline,
        actIndex,
      } = req.body || {};

      const actSchema =
        STORYLINE_SCHEMA
          .properties.acts.items;

      const response =
        await openai.responses.create({
          model: OPENAI_MODEL,

          input: [
            {
              role: "system",

              content:
                "You improve a single act for a children's fairytale storyline.",
            },

            {
              role: "user",

              content:
                `${prompt}\n\n` +
                `Storyline context:\n${JSON.stringify(
                  storyline,
                  null,
                  2
                )}\n\n` +
                `Regenerate ONLY Act ${actIndex + 1
                }.`,
            },
          ],

          text: {
            format: {
              type: "json_schema",

              name: "act",

              strict: true,

              schema: actSchema,
            },
          },
        });

      res.json({
        act: JSON.parse(
          response.output_text
        ),
      });
    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);

// =====================================================
// FULL FAIRYTALE
// =====================================================

//Turns the structured storyline into a complete children's fairytale while constraining the requested length to a safe word-count range.
app.post(
  "/api/fairytale",
  async (req, res) => {
    try {
      const {
        prompt,
        storyline,
        wordCount,
      } = req.body || {};

      let wc =
        Number(wordCount);

      if (
        !Number.isFinite(wc)
      ) {
        wc = 700;
      }

      wc = Math.max(
        250,
        Math.min(2000, wc)
      );

      const response =
        await openai.responses.create({
          model: OPENAI_MODEL,

          input: [
            {
              role: "system",

              content:
                "You write delightful children's fairytales.",
            },

            {
              role: "user",

              content:
                `${prompt}\n\n` +
                `Use this storyline:\n${JSON.stringify(
                  storyline,
                  null,
                  2
                )}\n\n` +
                `Write the FULL fairytale.\n` +
                `Length target: ${wc} words.`,
            },
          ],
        });

      res.json({
        text: response.output_text,
      });
    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);

// =====================================================
// NARRATION
// =====================================================

//Converts the generated storyline and fairytale into a narration script suitable for performing the story as children's shadow theatre.
app.post(
  "/api/narration",
  async (req, res) => {
    try {
      const {
        storyline,
        fairytaleText,
      } = req.body || {};

      const response =
        await openai.responses.create({
          model: OPENAI_MODEL,

          input: [
            {
              role: "system",

              content:
                "You are a playwright for children's shadow theatre.",
            },

            {
              role: "user",

              content:
                `Create a narration script.\n\n` +
                `Storyline:\n${JSON.stringify(
                  storyline,
                  null,
                  2
                )}\n\n` +
                `Story:\n${String(
                  fairytaleText ||
                  ""
                ).slice(0, 5000)}`,
            },
          ],
        });

      res.json({
        text: response.output_text,
      });
    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);

// =====================================================
// CHARACTER SIZE CLASSIFICATION
// =====================================================

//Classifies a character as adult-sized or child-sized so the frontend can scale the physical shadow-puppet appropriately.
app.post(
  "/api/character/classify-size",
  async (req, res) => {
    try {
      const {
        name,
        title,
        prompt,
      } = req.body || {};

      if (!name && !title && !prompt) {
        return res
          .status(400)
          .send(
            "Missing character information"
          );
      }

      const CHARACTER_SIZE_SCHEMA = {
        type: "object",
        additionalProperties: false,
        required: [
          "sizeClass",
          "reason"
        ],
        properties: {
          sizeClass: {
            type: "string",
            enum: [
              "adult",
              "child"
            ]
          },
          reason: {
            type: "string"
          }
        }
      };

      const response =
        await openai.responses.create({
          model: OPENAI_MODEL,

          input: [
            {
              role: "system",

              content:
                `You classify characters for physical shadow-puppet scaling.

Classify the character as exactly one of these categories:

- "child": a child, young boy, young girl, baby, young prince, young princess, pupil, or similarly child-sized character.
- "adult": an adult person or an adult-sized humanoid character, including a parent, teacher, queen, king, witch, wizard, elderly person, or grown character.

Use the character name, title, and description.

When age is not explicitly given, infer the most likely physical size from the description.

If the character is an animal, magical creature, or ambiguous non-human character, classify it according to the closest intended puppet scale:
- small or young creature -> child
- large, mature, or human-adult-sized creature -> adult.`
            },

            {
              role: "user",

              content:
                `Character name: ${String(
                  name || ""
                ).slice(0, 300)}

Character title: ${String(
                  title || ""
                ).slice(0, 300)}

Character description:
${String(
                  prompt || ""
                ).slice(0, 2000)}`
            }
          ],

          text: {
            format: {
              type: "json_schema",
              name:
                "character_size_classification",
              strict: true,
              schema:
                CHARACTER_SIZE_SCHEMA
            }
          }
        });

      const result = JSON.parse(
        response.output_text
      );

      res.json(result);

    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);


// =====================================================
// PUPPET PLAN
// =====================================================

// Analyses the completed story to create a deduplicated global character list and a plan of reusable character and background assets.
app.post(
  "/api/puppets/plan",
  async (req, res) => {
    try {
      const {
        storyline,
        fairytaleText,
      } = req.body || {};

      const actsCompact =
        storyline.acts.map(
          (a, i) => ({
            actIndex: i,
            title: a.title,
            summary: a.summary,
            beats: a.beats,
          })
        );

      const response =
        await openai.responses.create({
          model: OPENAI_MODEL,

          input: [
            {
              role: "system",

              content:
                "You design laser-cut shadow puppet assets.",
            },

            {
              role: "user",

              content:
                `We have a 5-act storyline:\n${JSON.stringify(
                  actsCompact,
                  null,
                  2
                )}\n\n` +
                `Optional fairytale text:\n${String(
                  fairytaleText ||
                  ""
                ).slice(0, 4000)}\n\n` +
                `IMPORTANT:\n` +
                `- Include every individually identifiable character who physically appears, speaks, acts, or participates in the finished story.\n` +
                `- Reuse characters globally whenever possible.\n` +
                `- Reuse backgrounds when the place is the same.\n` +
                `- Character prompts must remain visually consistent.\n` +

                `- Characters are SINGLE reusable puppet figures.\n` +
                `- NEVER generate scenes with multiple characters interacting.\n` +
                `- NEVER describe two people together.\n` +
                `- NEVER describe crowds.\n` +
                `- NEVER describe interaction.\n` +
                `- Each character asset must contain ONLY ONE isolated figure.\n` +

                `- Characters must have a SINGLE reusable version only; do NOT create variants.\n` +
                `- Puppet silhouettes must be simple and readable.\n` +
                `- Use high contrast between the puppet and the background.\n` +
                `- Puppet figures should never interact with any objects.\n` +
                `- Avoid tangled limbs.\n` +
                `- Avoid overlapping body parts.\n` +
                `- Prefer side profiles and strong silhouettes.\n\n` +

                `Output structure:\n` +
                `- characters[]: one global reusable list of unique characters in the whole story.\n` +
                `- Each character must appear ONLY ONCE in characters[].\n` +
                `- Do NOT create character variants for different acts, emotions, clothes, poses, ages, or situations.\n` +
                `- If the same character appears in several acts, reuse the same character key.\n\n` +

                `For each character output:\n` +
                `- name\n` +
                `- key\n` +
                `- prompt\n\n` +

                `For each act output:\n` +
                `- actSummaryShort\n` +
                `- background { key, prompt }\n` +
                `- characterKeys[] containing only keys from the global characters[] list.`,
            },
          ],

          text: {
            format: {
              type: "json_schema",

              name:
                "puppet_plan",

              strict: true,

              schema:
                PUPPET_PLAN_SCHEMA,
            },
          },
        });

      const plan = JSON.parse(
        response.output_text
      );

      // Normalize and sanitize global characters once
      const characterMap = new Map();

      for (const c of plan.characters || []) {
        try {
          const key = normalizeKey(c.name || c.key);

          if (!key || characterMap.has(key)) continue;

          const res = await sanitizeCharacterPrompt(
            String(c.prompt || "")
          );

          if (res && res.ok) {
            characterMap.set(key, {
              name: String(c.name || key),
              key,
              prompt: String(res.prompt || "")
            });
          } else {
            console.warn(
              "Character prompt failed validation and was omitted:",
              c.name
            );
          }
        } catch (err) {
          console.error(
            "Error sanitizing character prompt:",
            err
          );
        }
      }

      plan.characters = [...characterMap.values()];

      // Normalizes backgrounds and replaces characterKeys with full character objects.
      const backgroundMap = new Map();

      for (const act of plan.acts) {
        const bgPrompt = String(act.background.prompt || "")
          .trim()
          .toLowerCase();

        if (backgroundMap.has(bgPrompt)) {
          act.background.key = backgroundMap.get(bgPrompt);
        } else {
          const bgKey = normalizeKey(act.background.key);

          backgroundMap.set(bgPrompt, bgKey);

          act.background.key = bgKey;
        }

        const uniqueCharacterKeys = [
          ...new Set(
            (act.characterKeys || [])
              .map(k => normalizeKey(k))
              .filter(k => characterMap.has(k))
          )
        ];

        act.characters = uniqueCharacterKeys.map(k =>
          characterMap.get(k)
        );

        delete act.characterKeys;
      }
      res.json(plan);

    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);

// =====================================================
// IMAGE GENERATION
// =====================================================

//Generates or retrieves a reusable SVG character or background asset, using caching and shared pending promises to prevent unnecessary duplicate ComfyUI workflows.
app.post(
  "/api/puppets/image",
  async (req, res) => {
    try {
      const {
        kind,
        key,
        prompt,
        force,
      } = req.body || {};

      if (
        !kind ||
        !key ||
        !prompt
      ) {
        return res
          .status(400)
          .send(
            "Missing kind/key/prompt"
          );
      }

      const normKey =
        normalizeKey(key);

      // =========================================
      // RETURN CACHE
      // =========================================

      if (
        !force &&
        imageCache.has(normKey)
      ) {
        return res.json(
          imageCache.get(normKey)
        );
      }

      // =========================================
      // RETURN PENDING GENERATION
      // =========================================

      if (
        pendingGenerations.has(
          normKey
        )
      ) {
        const existing =
          await pendingGenerations.get(
            normKey
          );

        return res.json(existing);
      }

      // =========================================
      // PROMPTS
      // =========================================

      const hardRules =
        `ABSOLUTE RULES:\n` +
        `- ONLY pure black shapes on pure white background\n` +
        `- NO gradients\n` +
        `- NO grey\n` +
        `- NO anti-aliasing\n` +
        `- NO shading\n` +
        `- NO textures\n` +
        `- NO shadows\n` +
        `- NO transparency\n` +
        `- NO text\n` +
        `- Thick connected silhouette shapes\n` +
        `- Strong readable outer contours\n` +
        `- Suitable for laser cutting\n` +
        `- High contrast stencil style\n`;

      const kindRules =
        kind ===
          "background"
          ? `BACKGROUND: theatrical shadow puppet stage background.\n` +
          `- CENTRAL AREA: The middle 50-60% of the composition MUST be completely empty and unoccupied.\n` +
          `- EDGES ONLY: All decorative elements, scenery, and details must be positioned ONLY at the edges and corners.\n` +
          `- STAGE FLOOR: Keep the center clear for character puppets to appear and move.\n` +
          `- NO CENTER OBJECTS: Absolutely NO trees, buildings, rocks, or objects in the central area.\n` +
          `- FRAME DESIGN: Design the background as a theatrical frame with decorative borders, leaving the stage center empty for puppet interaction.\n`
          : `CHARACTER RULES:
           - ONE SINGLE character ONLY
           - NEVER multiple people
           - NEVER interaction
           - NEVER duplicated figures
           - NEVER scene composition
           - ONLY ONE isolated puppet figure
           - Full body visible from head to feet
           - Centered composition
           - Strong readable silhouette
           - Side profile preferred
           - Laser-cut shadow puppet style
           - Connected black silhouette
           - No floating detached pieces
           - No background
           - No scenery
           - No environment
           - No additional creatures
           - No extra limbs
           - No duplicate heads
           - Puppet design must be reusable across scenes
           - White empty background
           `;

      const positivePrompt =
        hardRules +
        "\n" +
        kindRules +
        "\n" +
        prompt;

      const negativePrompt =
        "multiple people, two characters, crowd, interaction, duplicated body, duplicate person, extra limbs, extra arms, extra legs, extra heads, background scene, environment, scenery, color, gradients, grey, shading, shadows, texture, text, watermark, blur";

      const workflowPath =
        kind ===
          "background"
          ? requireEnv(
            "BACKGROUND_WORKFLOW_PATH"
          )
          : requireEnv(
            "CHARACTER_WORKFLOW_PATH"
          );

      // =========================================
      // GENERATION
      // =========================================

      //Workflow execution
      const generationPromise =
        (async () => {

          //Generates a new seed
          const generationSeed =
            Math.floor(
              Math.random() *
              Number.MAX_SAFE_INTEGER
            );

          console.log(
            `Generating ${kind} "${normKey}" with seed:`,
            generationSeed
          );

          //Runs the selected ComfyUI workflow
          const svgResult =
            await runPromptWorkflow({
              workflowPath,
              positivePrompt,
              negativePrompt,
              seed: generationSeed
            });

          const record = {
            key: normKey,
            kind,
            seed: generationSeed,

            svgText:
              svgResult.svgText,

            dataUrl:
              svgResult.dataUrl,

            mimeType:
              svgResult.mimeType,

            createdAt:
              new Date().toISOString(),

            source:
              "comfy_svg"
          };

          imageCache.set(
            normKey,
            record
          );

          return record;

        })();

      pendingGenerations.set(
        normKey,
        generationPromise
      );

      try {
        const record =
          await generationPromise;

        res.json(record);

      } finally {
        pendingGenerations.delete(
          normKey
        );
      }


    } catch (e) {
      console.error(e);

      res
        .status(500)
        .send(
          String(
            e?.message || e
          )
        );
    }
  }
);

// =====================================================
// START
// =====================================================

//Starts the Express server on the configured port and reports the local server address when it is ready to receive requests.
app.listen(PORT, () => {
  console.log(
    `Server running at http://localhost:${PORT}`
  );
});