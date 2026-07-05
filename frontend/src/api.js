const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`)
    if (!res.ok) return false
    const data = await res.json()
    return data.status === 'ok'
  } catch {
    return false
  }
}

// Sends {country, year, hicp, unemployment_rate} and returns {hpi}
export async function predictHpi(payload) {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  const data = await res.json().catch(() => null)

  if (!res.ok) {
    const message = data && data.error ? data.error : `Request failed (${res.status})`
    throw new Error(message)
  }

  return data
}
