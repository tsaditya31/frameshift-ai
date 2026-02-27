// SSE Chat client utilities (shared across pages)
// Page-specific chat logic is inline in each template.

// Shared utility: auto-resize textareas
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('textarea').forEach(ta => {
    ta.addEventListener('input', () => {
      ta.style.height = 'auto';
      ta.style.height = ta.scrollHeight + 'px';
    });
  });

  // Upload zone drag-and-drop
  const zone = document.getElementById('drop-zone');
  if (zone) {
    zone.addEventListener('click', () => zone.querySelector('input')?.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--accent)'; });
    zone.addEventListener('dragleave', () => zone.style.borderColor = '');
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.style.borderColor = '';
      const input = zone.querySelector('input');
      if (input) {
        const dt = new DataTransfer();
        for (const f of e.dataTransfer.files) dt.items.add(f);
        input.files = dt.files;
        const label = zone.querySelector('p');
        if (label) label.textContent = `${e.dataTransfer.files.length} file(s) selected`;
      }
    });
  }

  // Submit chat on Ctrl+Enter / Cmd+Enter
  const chatInput = document.getElementById('chat-input');
  if (chatInput) {
    chatInput.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('chat-form')?.dispatchEvent(new Event('submit'));
      }
    });
  }
});
