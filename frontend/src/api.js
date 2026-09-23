// Single place that knows how to talk to Flask. In dev, Vite proxies /api to
// http://127.0.0.1:5000 (see vite.config.js); in production Flask serves this
// bundle itself, so the same relative paths work unchanged.

async function request(path, options = {}) {
  const response = await fetch(path, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.error || `${path} failed (${response.status})`)
  return data
}

export const getModel = () => request('/api/model')
export const getAnalytics = () => request('/api/analytics')
export const refresh = () => request('/api/refresh', { method: 'POST' })

export const classify = (text) =>
  request('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
