// 「Agent 设计问答」前端：调用 /api/chat，渲染答案与引用出处。
(() => {
  const messagesEl = document.getElementById('messages');
  const form = document.getElementById('composer');
  const input = document.getElementById('question');
  const sendBtn = document.getElementById('send');
  const sourcesEl = document.getElementById('sources');
  const sourcesEmpty = document.getElementById('sources-empty');
  const badgeRoute = document.getElementById('badge-route');
  const badgeMeta = document.getElementById('badge-meta');

  const ROUTE_LABEL = {
    design: '设计模式库',
    deep: '深入理解库',
    both: '双库检索',
  };

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // 轻量 Markdown：标题 / 列表 / 粗体 / 行内代码 / 代码块 / 引用角标
  function renderMarkdown(text) {
    const blocks = [];
    let src = escapeHtml(text).replace(/```([\s\S]*?)```/g, (_, code) => {
      blocks.push(`<pre><code>${code.trim()}</code></pre>`);
      return `\u0000BLOCK${blocks.length - 1}\u0000`;
    });

    src = src
      .replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
      .replace(/^##\s+(.+)$/gm, '<h3>$1</h3>')
      .replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>[\s\S]*?<\/li>)(?!\s*<li>)/g, '<ul>$1</ul>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');

    src = src
      .split(/\n{2,}/)
      .map((chunk) => {
        const trimmed = chunk.trim();
        if (!trimmed) return '';
        if (/^<(h3|ul|pre)/.test(trimmed) || /^\u0000BLOCK/.test(trimmed)) return trimmed;
        return `<p>${trimmed.replace(/\n/g, '<br/>')}</p>`;
      })
      .join('');

    return src.replace(/\u0000BLOCK(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
  }

  // 答案正文与服务端的“引用出处”清单分离：正文进气泡，引用进右侧面板
  function splitAnswer(text) {
    const marker = text.search(/\n#{2,4}\s*引用出处|\n\*\*引用出处\*\*/);
    if (marker > 0) return text.slice(0, marker);
    return text;
  }

  function addMessage(role, html, extraHtml = '') {
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;
    wrap.innerHTML = `<div class="bubble">${extraHtml}${html}</div>`;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  function setRoute(route, latency, usedLlm) {
    const key = route.route || 'both';
    badgeRoute.className = `badge ${key}`;
    badgeRoute.textContent = ROUTE_LABEL[key] || key;
    const conf = typeof route.confidence === 'number' ? route.confidence.toFixed(2) : '-';
    const src = { llm: 'LLM 判定', rule: '规则兜底', override: '显式指定' }[route.source] || route.source;
    badgeMeta.textContent = `置信度 ${conf} · ${src} · ${latency}ms · ${usedLlm ? 'LLM 生成' : '检索摘要'}`;
  }

  function renderSources(citations) {
    sourcesEl.innerHTML = '';
    if (!citations || citations.length === 0) {
      sourcesEmpty.textContent = '本轮无引用（知识库未覆盖或尚未入库）。';
      sourcesEmpty.style.display = 'block';
      return;
    }
    sourcesEmpty.style.display = 'none';
    citations.forEach((c) => {
      const el = document.createElement('div');
      el.className = 'source';
      el.innerHTML = `
        <div><span class="idx">${c.index}</span><span class="book">《${escapeHtml(c.book_title)}》</span></div>
        <div class="loc">${c.chapter_no ? `第 ${c.chapter_no} 章` : '前言/附录'} ${escapeHtml(c.chapter_title || '')}${
        c.section_no ? ` · ${escapeHtml(c.section_no)} ${escapeHtml(c.section_title || '')}` : ''
      }</div>
        <div class="loc">第 ${c.page} 页 · 第 ${c.para_start}${c.para_end !== c.para_start ? '–' + c.para_end : ''} 段 · ${escapeHtml(c.chunk_type)}</div>
        <div class="excerpt">${escapeHtml(c.excerpt)}</div>`;
      sourcesEl.appendChild(el);
    });
  }

  // 同源 + 相对路径：前端可挂在任意子路径（如 /agent-book/），API 由 nginx 转发到后端。
  // 要求页面 URL 以 / 结尾（nginx 对 /agent-book 做 301 到 /agent-book/）。
  const API_CHAT = 'api/chat';
  const API_HEALTH = 'api/health';

  async function ask(question) {
    if (!question.trim()) return;
    addMessage('user', `<p>${escapeHtml(question)}</p>`);
    input.value = '';
    sendBtn.disabled = true;
    badgeRoute.className = 'badge';
    badgeRoute.textContent = '思考中…';
    badgeMeta.textContent = '';

    const pending = addMessage('assistant', '<p class="muted">正在路由并检索…</p>');

    try {
      const resp = await fetch(API_CHAT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`);
      }

      const routeBadges = `<div class="badge-line">
        <span class="badge ${data.route.route}">${ROUTE_LABEL[data.route.route] || data.route.route}</span>
        <span class="badge">${data.used_llm ? 'LLM 生成' : '检索摘要'}</span>
        ${data.route.clarify_hint ? '<span class="badge">建议澄清</span>' : ''}
      </div>`;
      const clarify = data.route.clarify_hint
        ? `<p class="muted small">提示：${escapeHtml(data.route.clarify_hint)}</p>`
        : '';
      pending.remove();
      addMessage('assistant', renderMarkdown(splitAnswer(data.answer)), routeBadges + clarify);
      renderSources(data.citations);
      setRoute(data.route, data.latency_ms, data.used_llm);
    } catch (err) {
      pending.remove();
      addMessage('assistant', `<p>请求失败：${escapeHtml(String(err.message || err))}</p>`);
      badgeRoute.className = 'badge err';
      badgeRoute.textContent = '出错';
      badgeMeta.textContent = '';
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    ask(input.value);
  });

  document.querySelectorAll('.sample').forEach((btn) => {
    btn.addEventListener('click', () => ask(btn.textContent));
  });

  // 健康状态
  fetch(API_HEALTH)
    .then((r) => r.json())
    .then((h) => {
      if (h.status !== 'ok') {
        badgeRoute.className = 'badge err';
        badgeRoute.textContent = '服务降级';
        badgeMeta.textContent = h.collection_ready ? 'Milvus 就绪，数据可能为空' : 'Milvus 未就绪，请先 ingest';
      }
    })
    .catch(() => {});
})();
