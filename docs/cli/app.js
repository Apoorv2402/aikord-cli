/**
 * aikord-cli Web Developer Documentation — Application Logic
 * Interactive Terminal Playground, Visual Configurator, Live Search, and UI Helpers.
 */

document.addEventListener('DOMContentLoaded', () => {
  initSearch();
  initCodeTabs();
  initCopyButtons();
  initSidebar();
  initPlayground();
  initConfigurator();
  initScrollSpy();
});

/* ==========================================================================
   1. Search Modal (Ctrl+K / Cmd+K)
   ========================================================================== */

const SEARCH_INDEX = [
  { title: "Introduction & Overview", category: "Getting Started", anchor: "#overview", snippet: "Open-source, subscription-free AI terminal copilot with local SLMs." },
  { title: "Quickstart (5-Minute Setup)", category: "Getting Started", anchor: "#quickstart", snippet: "Install via git clone, pip install -e ., and start running queries." },
  { title: "Local SLM Setup (Ollama)", category: "Getting Started", anchor: "#ollama-setup", snippet: "Run offline with qwen2.5-coder:7b using Ollama at localhost:11434." },
  { title: "Cloud Setup (DeepSeek / Groq / OpenAI)", category: "Getting Started", anchor: "#cloud-setup", snippet: "Configure API keys for DeepSeek, Groq Llama 3.3, and OpenAI." },
  { title: "Terminal RAG & Context Injection", category: "Architecture", anchor: "#context-injection", snippet: "Automatic scraping of OS, Shell dialect, CWD, and Git branch/status." },
  { title: "DuckDB VSS Semantic Cache", category: "Architecture", anchor: "#semantic-cache", snippet: "HNSW vector indexing and all-MiniLM-L6-v2 embeddings for sub-10ms hits." },
  { title: "Dynamic Complexity Router", category: "Architecture", anchor: "#complexity-router", snippet: "Heuristic MoE scoring (tokens, pipes, destructive ops, git repo state)." },
  { title: "3-Layer Defense-in-Depth Safety", category: "Architecture", anchor: "#safety-architecture", snippet: "Prompt contracts, Pydantic validation, and deterministic pattern checks." },
  { title: "ReAct Self-Healing Loop", category: "Architecture", anchor: "#self-healing", snippet: "Autonomous diagnostic retry loop with cycle detection and safety guards." },
  { title: "Telemetry & Observability", category: "Architecture", anchor: "#telemetry", snippet: "Append-only JSONL event logging for Latency, TTFT, and token cost." },
  { title: "aikord suggest", category: "CLI Reference", anchor: "#cmd-suggest", snippet: "Main command: natural language to shell command generation." },
  { title: "aikord explain", category: "CLI Reference", anchor: "#cmd-explain", snippet: "Flag-by-flag plain-English explanation of any shell command." },
  { title: "aikord cache", category: "CLI Reference", anchor: "#cmd-cache", snippet: "Inspect cache hit stats and clear vector database entries." },
  { title: "aikord config", category: "CLI Reference", anchor: "#cmd-config", snippet: "View and set runtime configuration keys in config.json." },
  { title: "aikord plugin", category: "CLI Reference", anchor: "#cmd-plugin", snippet: "List discovered and loaded plugins from config directories." },
  { title: "aikord eval", category: "CLI Reference", anchor: "#cmd-eval", snippet: "Benchmark active provider against the 50-example ground-truth dataset." },
  { title: "Interactive Terminal Playground", category: "Interactive", anchor: "#playground", snippet: "Simulate live context gathering, routing, and self-healing execution." },
  { title: "Visual Config Generator", category: "Interactive", anchor: "#configurator", snippet: "Configure thresholds, models, and export config.json or .env." },
  { title: "Plugin Development Guide", category: "Plugins", anchor: "#plugins", snippet: "Subclass PluginBase and hook into pre/post suggest and execution." },
  { title: "Benchmarking & Evaluation Harness", category: "Evals", anchor: "#evals", snippet: "Measure JSON parse rate, destructive recall (100%), and p50/p95 latency." },
  { title: "Troubleshooting & FAQ", category: "Help", anchor: "#faq", snippet: "Resolving Ollama connectivity, DuckDB VSS installation, and permissions." },
];

function initSearch() {
  const backdrop = document.getElementById('search-modal-backdrop');
  const openBtn = document.getElementById('quick-search-trigger');
  const input = document.getElementById('search-modal-input');
  const resultsContainer = document.getElementById('search-results');

  if (!backdrop || !input) return;

  function openModal() {
    backdrop.style.display = 'flex';
    input.value = '';
    renderResults(SEARCH_INDEX);
    input.focus();
  }

  function closeModal() {
    backdrop.style.display = 'none';
  }

  if (openBtn) openBtn.addEventListener('click', openModal);

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      backdrop.style.display === 'flex' ? closeModal() : openModal();
    }
    if (e.key === 'Escape' && backdrop.style.display === 'flex') {
      closeModal();
    }
  });

  backdrop.addEventListener('click', (e) => {
    if (e.target === backdrop) closeModal();
  });

  input.addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    if (!q) {
      renderResults(SEARCH_INDEX);
      return;
    }
    const filtered = SEARCH_INDEX.filter(item => 
      item.title.toLowerCase().includes(q) ||
      item.category.toLowerCase().includes(q) ||
      item.snippet.toLowerCase().includes(q)
    );
    renderResults(filtered);
  });

  function renderResults(items) {
    if (!items.length) {
      resultsContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-muted);">No matching topics found.</div>';
      return;
    }
    resultsContainer.innerHTML = items.map((item, idx) => `
      <a href="${item.anchor}" class="search-result-item ${idx === 0 ? 'selected' : ''}" onclick="document.getElementById('search-modal-backdrop').style.display='none'">
        <div class="result-category">${item.category}</div>
        <div class="result-title">${item.title}</div>
        <div class="result-snippet">${item.snippet}</div>
      </a>
    `).join('');
  }
}

/* ==========================================================================
   2. Code Tabs & Copy-to-Clipboard
   ========================================================================== */

function initCodeTabs() {
  document.querySelectorAll('.code-window').forEach(windowEl => {
    const tabs = windowEl.querySelectorAll('.code-tab-btn');
    const panes = windowEl.querySelectorAll('.code-pane');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetId = tab.getAttribute('data-tab');
        tabs.forEach(t => t.classList.remove('active'));
        panes.forEach(p => p.style.display = 'none');

        tab.classList.add('active');
        const activePane = windowEl.querySelector(`.code-pane[data-pane="${targetId}"]`);
        if (activePane) activePane.style.display = 'block';
      });
    });
  });
}

function initCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetSelector = btn.getAttribute('data-copy-target');
      let text = '';
      if (targetSelector) {
        const target = document.querySelector(targetSelector);
        text = target ? target.innerText : '';
      } else {
        const codeEl = btn.closest('.code-window, .hero-quick-install')?.querySelector('pre, code, .cmd-text');
        text = codeEl ? codeEl.innerText.trim() : '';
      }

      if (!text) return;

      navigator.clipboard.writeText(text).then(() => {
        const originalHtml = btn.innerHTML;
        btn.innerHTML = '<span>✓ Copied!</span>';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.innerHTML = originalHtml;
          btn.classList.remove('copied');
        }, 1800);
      });
    });
  });
}

/* ==========================================================================
   3. Sidebar Navigation & Mobile Drawer
   ========================================================================== */

function initSidebar() {
  const toggleBtn = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('docs-sidebar');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
      if (window.innerWidth <= 900 && 
          !sidebar.contains(e.target) && 
          !toggleBtn.contains(e.target) && 
          sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
      }
    });
  }
}

function initScrollSpy() {
  const sections = document.querySelectorAll('section[id], div[id].doc-section');
  const sidebarLinks = document.querySelectorAll('.sidebar-link');
  const tocLinks = document.querySelectorAll('.toc-link');

  window.addEventListener('scroll', () => {
    let currentId = '';
    const scrollPos = window.scrollY + 120;

    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = section.getAttribute('id');
      }
    });

    if (currentId) {
      sidebarLinks.forEach(link => {
        const href = link.getAttribute('href');
        link.classList.toggle('active', href === `#${currentId}`);
      });
      tocLinks.forEach(link => {
        const href = link.getAttribute('href');
        link.classList.toggle('active', href === `#${currentId}`);
      });
    }
  });
}

/* ==========================================================================
   4. Interactive Component 1: Terminal Simulation & Playground
   ========================================================================== */

const PLAYGROUND_PRESETS = {
  safe: {
    query: "list running docker containers sorted by memory usage",
    shell: "bash",
    command: "docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' | sort -k 2 -h",
    explanation: "docker stats: stream container resource usage. --no-stream: take a single snapshot. format: display table with memory and cpu. sort: sort lines numerically by column 2.",
    is_destructive: false,
    risk_level: "low",
    alternatives: ["docker ps -s", "ctop"],
    score: 0.28,
    is_cache_hit: true,
    cache_similarity: 0.94,
    provider: "cache",
    fails: false,
    stdout: "CONTAINER ID   NAME          MEM USAGE / LIMIT     CPU %\n4c01db0b339c   postgres-db   142.4MiB / 7.661GiB   0.12%\nd723820a23ff   redis-cache   32.1MiB / 7.661GiB    0.04%"
  },
  pipeline: {
    query: "find python files modified in last 7 days and count total lines of code",
    shell: "bash",
    command: "find . -name '*.py' -mtime -7 | xargs wc -l | sort -n",
    explanation: "find: search directory hierarchy for .py files modified within 7 days. xargs wc -l: pass found paths to wc to count lines. sort -n: sort totals numerically.",
    is_destructive: false,
    risk_level: "low",
    alternatives: ["git diff --stat '@{7 days ago}'", "cloc --diff"],
    score: 0.55,
    is_cache_hit: false,
    provider: "deepseek",
    fails: false,
    stdout: "  120 ./src/config.py\n  216 ./src/engine/router.py\n  338 ./src/cache/vector_cache.py\n  584 ./src/cli.py\n 1258 total"
  },
  destructive: {
    query: "delete node_modules directory and clean build cache",
    shell: "bash",
    command: "rm -rf ./node_modules ./dist ./.cache",
    explanation: "rm: remove files or directories. -r: remove directories and their contents recursively. -f: ignore nonexistent files and arguments, never prompt.",
    is_destructive: true,
    risk_level: "critical",
    alternatives: ["trash-put ./node_modules ./dist", "git clean -fdX"],
    score: 0.65,
    is_cache_hit: false,
    provider: "deepseek",
    fails: false,
    stdout: "Removed 1,420 files across 3 directories (reclaimed 340 MB)."
  },
  healing: {
    query: "kill process currently listening on port 8080",
    shell: "bash",
    command: "kill $(lsof -t -i:8080)",
    explanation: "kill: send termination signal to PID. lsof -t -i:8080: extract raw PID holding TCP port 8080.",
    is_destructive: true,
    risk_level: "high",
    alternatives: ["fuser -k 8080/tcp", "npx kill-port 8080"],
    score: 0.52,
    is_cache_hit: false,
    provider: "deepseek",
    fails: true,
    initial_stderr: "kill: usage: kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]\nlsof: command requires root permission to inspect socket",
    healed_command: "fuser -k 8080/tcp || sudo kill -9 $(sudo lsof -t -i:8080)",
    healed_explanation: "Replaced with fuser -k which directly signals the socket owner, with a sudo lsof fallback for elevated privileges.",
    stdout: "8080/tcp: 38241\nProcess 38241 terminated cleanly."
  }
};

let currentSimulation = PLAYGROUND_PRESETS.safe;

function initPlayground() {
  const queryInput = document.getElementById('pg-query');
  const shellSelect = document.getElementById('pg-shell');
  const providerSelect = document.getElementById('pg-provider');
  const submitBtn = document.getElementById('pg-submit-btn');
  const presetChips = document.querySelectorAll('.preset-chip');

  if (!queryInput || !submitBtn) return;

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const presetKey = chip.getAttribute('data-preset');
      const preset = PLAYGROUND_PRESETS[presetKey];
      if (preset) {
        queryInput.value = preset.query;
        shellSelect.value = preset.shell;
        runSimulation(preset);
      }
    });
  });

  submitBtn.addEventListener('click', () => {
    const q = queryInput.value.trim();
    if (!q) return;

    // Detect if custom query matches any preset or simulate custom
    let match = null;
    for (const key of Object.keys(PLAYGROUND_PRESETS)) {
      if (PLAYGROUND_PRESETS[key].query.toLowerCase() === q.toLowerCase()) {
        match = PLAYGROUND_PRESETS[key];
        break;
      }
    }

    if (match) {
      runSimulation(match);
    } else {
      // Generate synthetic simulation for custom query
      const isDestr = /rm|del|drop|kill|clean|format/i.test(q);
      const pipeCount = (q.match(/\||&|;/g) || []).length;
      const score = Math.min(1.0, (q.split(' ').length / 50) * 0.3 + (pipeCount / 4) * 0.25 + (isDestr ? 0.3 : 0));
      const forcedProvider = providerSelect.value;
      const provider = forcedProvider !== 'auto' ? forcedProvider : (score >= 0.4 ? 'deepseek' : 'ollama');

      const customSim = {
        query: q,
        shell: shellSelect.value,
        command: shellSelect.value === 'powershell' ? `Get-ChildItem -Recurse | Select-String -Pattern "${q.split(' ')[0]}"` : `grep -rnI "${q.split(' ')[0]}" .`,
        explanation: `Synthesized suggestion for: ${q}. Shell dialect: ${shellSelect.value}.`,
        is_destructive: isDestr,
        risk_level: isDestr ? 'high' : 'low',
        alternatives: ["ag -i", "rg"],
        score: parseFloat(score.toFixed(2)),
        is_cache_hit: false,
        provider: provider,
        fails: false,
        stdout: "Command executed successfully in demo environment."
      };
      runSimulation(customSim);
    }
  });

  // Run default
  runSimulation(PLAYGROUND_PRESETS.safe);
}

function runSimulation(simData) {
  currentSimulation = simData;
  const viewport = document.getElementById('terminal-viewport');
  if (!viewport) return;

  // Render suggestion panel
  const isDestr = simData.is_destructive;
  const borderClass = isDestr ? 'destructive' : '';
  const panelTitle = isDestr 
    ? '⚠ DESTRUCTIVE COMMAND — Review Carefully' 
    : '🤖 aikord-cli Suggestion';

  const riskClass = simData.risk_level;
  const sourceBadge = simData.is_cache_hit 
    ? `⚡ Cache Hit <span style="opacity:0.6;">(similarity: ${simData.cache_similarity})</span>` 
    : `🤖 ${simData.provider} <span style="opacity:0.6;">(Complexity: ${simData.score})</span>`;

  viewport.innerHTML = `
    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 8px;">
      [CWD: /home/user/project] &nbsp;|&nbsp; [Shell: ${simData.shell}] &nbsp;|&nbsp; [Git: yes | branch: main]
    </div>
    
    <div class="rich-panel ${borderClass}">
      <div class="rich-panel-title">${panelTitle}</div>
      <div class="rich-command-text">${escapeHtml(simData.command)}</div>
      
      <div class="rich-meta-row">
        <span class="badge-source">${sourceBadge}</span>
        <span class="badge-risk ${riskClass}">⚑ ${simData.risk_level.toUpperCase()}</span>
        <span style="color: var(--text-muted); margin-left: auto;">${simData.is_cache_hit ? '4ms' : '230ms'}</span>
      </div>
    </div>

    <div style="font-size: 0.86rem; color: var(--text-secondary); font-style: italic; margin-bottom: 12px;">
      ${escapeHtml(simData.explanation)}
    </div>

    <div class="interactive-menu-strip">
      <button class="menu-choice-btn primary" id="btn-tui-run">▶ [R]un</button>
      <button class="menu-choice-btn" id="btn-tui-edit">✏ [E]dit</button>
      <button class="menu-choice-btn" id="btn-tui-explain">📖 [e]Xplain</button>
      <button class="menu-choice-btn" id="btn-tui-abort">✖ [A]bort</button>
    </div>

    <div id="tui-output-area" style="margin-top: 18px;"></div>
    <div id="healer-stepper" class="healer-stepper"></div>
  `;

  // Attach interactive button listeners
  document.getElementById('btn-tui-run').addEventListener('click', handleTuiRun);
  document.getElementById('btn-tui-edit').addEventListener('click', handleTuiEdit);
  document.getElementById('btn-tui-explain').addEventListener('click', handleTuiExplain);
  document.getElementById('btn-tui-abort').addEventListener('click', handleTuiAbort);
}

function handleTuiRun() {
  const out = document.getElementById('tui-output-area');
  const stepper = document.getElementById('healer-stepper');
  if (!out) return;

  if (!currentSimulation.fails) {
    out.innerHTML = `
      <div style="color: var(--accent-green); font-weight: 600; margin-bottom: 6px;">
        ✅ Success <span style="color: var(--text-muted); font-size: 0.8rem;">(14ms)</span>
      </div>
      <div style="background: rgba(0,0,0,0.5); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.82rem; white-space: pre-wrap;">${escapeHtml(currentSimulation.stdout)}</div>
    `;
    return;
  }

  // Failing simulation with Self-Healing ReAct Demo
  out.innerHTML = `
    <div style="color: var(--accent-red); font-weight: 600; margin-bottom: 4px;">
      ❌ Failed <span style="color: var(--text-muted); font-size: 0.8rem;">(exit 1, 18ms)</span>
    </div>
    <div style="background: rgba(239,68,68,0.08); color: #fca5a5; padding: 10px; border-radius: 6px; border: 1px solid rgba(239,68,68,0.25); font-size: 0.8rem; margin-bottom: 12px; white-space: pre-wrap;">${escapeHtml(currentSimulation.initial_stderr)}</div>
  `;

  stepper.style.display = 'block';
  stepper.innerHTML = `
    <div style="font-weight: 700; color: var(--accent-yellow); margin-bottom: 8px;">
      🔄 ReAct Self-Healing Loop Activated (--auto-fix)
    </div>
    <div class="step-indicator" id="step-1">⏳ Step 1: REASON — Submitting stderr & context to LLM...</div>
    <div class="step-indicator" id="step-2" style="opacity: 0.4;">⏳ Step 2: ACT — Proposing corrected command...</div>
    <div class="step-indicator" id="step-3" style="opacity: 0.4;">⏳ Step 3: OBSERVE — Executing healed command...</div>
  `;

  setTimeout(() => {
    const s1 = document.getElementById('step-1');
    if (s1) s1.innerHTML = '✔ Step 1: REASON — LLM diagnosed permission socket mismatch.';
    const s2 = document.getElementById('step-2');
    if (s2) {
      s2.style.opacity = '1';
      s2.innerHTML = `✔ Step 2: ACT — Fixed: <code style="color:var(--accent-cyan);">${escapeHtml(currentSimulation.healed_command)}</code>`;
    }
  }, 900);

  setTimeout(() => {
    const s3 = document.getElementById('step-3');
    if (s3) {
      s3.style.opacity = '1';
      s3.innerHTML = '✔ Step 3: OBSERVE — Exit code 0! Success recorded to cache.';
    }
    out.innerHTML += `
      <div style="margin-top: 14px; color: var(--accent-green); font-weight: 600;">
        ✅ Self-healing succeeded (after 1 retry)
      </div>
      <div style="background: rgba(0,0,0,0.5); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.82rem; margin-top: 6px; white-space: pre-wrap;">${escapeHtml(currentSimulation.stdout)}</div>
    `;
  }, 1900);
}

function handleTuiEdit() {
  const newCmd = prompt("Edit command before running:", currentSimulation.command);
  if (newCmd && newCmd !== currentSimulation.command) {
    currentSimulation.command = newCmd;
    runSimulation(currentSimulation);
  }
}

function handleTuiExplain() {
  const out = document.getElementById('tui-output-area');
  if (!out) return;
  out.innerHTML = `
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 14px;">
      <div style="color: var(--accent-yellow); font-weight: 700; margin-bottom: 6px;">📖 Flag-by-Flag Command Breakdown</div>
      <div style="font-size: 0.85rem; line-height: 1.5; color: #f3f4f6;">${escapeHtml(currentSimulation.explanation)}</div>
    </div>
  `;
}

function handleTuiAbort() {
  const out = document.getElementById('tui-output-area');
  if (!out) return;
  out.innerHTML = `<div style="color: var(--text-muted); font-style: italic; padding: 8px 0;">Aborted.</div>`;
}

/* ==========================================================================
   5. Interactive Component 2: Visual Configurator
   ========================================================================== */

function initConfigurator() {
  const slCache = document.getElementById('cfg-cache-thresh');
  const slRouting = document.getElementById('cfg-routing-thresh');
  const slRetries = document.getElementById('cfg-max-retries');
  const slTimeout = document.getElementById('cfg-timeout');
  const selProvider = document.getElementById('cfg-provider');
  const selCloud = document.getElementById('cfg-cloud-provider');
  const chkAutoFix = document.getElementById('cfg-autofix');

  const jsonOut = document.getElementById('cfg-json-output');
  const envOut = document.getElementById('cfg-env-output');

  if (!slCache || !jsonOut) return;

  function update() {
    const cacheVal = parseFloat(slCache.value).toFixed(2);
    const routingVal = parseFloat(slRouting.value).toFixed(2);
    const retriesVal = parseInt(slRetries.value, 10);
    const timeoutVal = parseInt(slTimeout.value, 10);
    const providerVal = selProvider.value;
    const cloudVal = selCloud.value;
    const autoFixVal = chkAutoFix.checked;

    document.getElementById('lbl-cache-thresh').innerText = cacheVal;
    document.getElementById('lbl-routing-thresh').innerText = routingVal;
    document.getElementById('lbl-max-retries').innerText = retriesVal;
    document.getElementById('lbl-timeout').innerText = `${timeoutVal}s`;

    const configObj = {
      provider: providerVal,
      model: providerVal === "ollama" ? "qwen2.5-coder:7b" : "default",
      endpoint: "http://localhost:11434/v1",
      api_key: "ollama",
      cloud_provider: cloudVal,
      cloud_model: cloudVal === "deepseek" ? "deepseek-chat" : (cloudVal === "groq" ? "llama-3.3-70b-versatile" : "gpt-4o-mini"),
      cloud_endpoint: cloudVal === "deepseek" ? "https://api.deepseek.com/v1" : (cloudVal === "groq" ? "https://api.groq.com/openai/v1" : "https://api.openai.com/v1"),
      cloud_api_key: "sk-...",
      routing_threshold: parseFloat(routingVal),
      cache_threshold: parseFloat(cacheVal),
      cache_db_path: "~/.config/aikord-cli/cache.duckdb",
      max_retries: retriesVal,
      auto_fix: autoFixVal,
      request_timeout: timeoutVal
    };

    jsonOut.innerText = JSON.stringify(configObj, null, 2);

    envOut.innerText = [
      `# aikord-cli environment variable overrides`,
      `export AIKORD_PROVIDER="${providerVal}"`,
      `export AIKORD_CLOUD_PROVIDER="${cloudVal}"`,
      `export AIKORD_CLOUD_API_KEY="sk-your-key-here"`,
      `export AIKORD_AUTO_FIX="${autoFixVal}"`,
      `export AIKORD_MAX_RETRIES="${retriesVal}"`
    ].join('\n');
  }

  [slCache, slRouting, slRetries, slTimeout, selProvider, selCloud, chkAutoFix].forEach(el => {
    if (el) el.addEventListener('input', update);
  });

  update();
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}
