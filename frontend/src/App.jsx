import { useEffect, useState } from 'react'
import PredictionForm from './components/PredictionForm.jsx'
import IndexCard from './components/IndexCard.jsx'
import HistoryTable from './components/HistoryTable.jsx'
import { checkHealth, predictHpi } from './api.js'

const STORAGE_KEY = 'hpi-prediction-history'

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export default function App() {
  const [tab, setTab] = useState('predict')
  const [online, setOnline] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState(loadHistory)

  useEffect(() => {
    checkHealth().then(setOnline)
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
  }, [history])

  async function handlePredict(payload) {
    setSubmitting(true)
    setError(null)
    try {
      const result = await predictHpi(payload)
      const entry = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        ...payload,
        hpi: result.hpi,
      }
      setHistory((prev) => [entry, ...prev])
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  function handleClear() {
    setHistory([])
  }

  const latest = history[0] ?? null

  return (
    <div className="app-shell">
      <header className="masthead">
        <div>
          <h1 className="masthead-title">HPI Index Bulletin</h1>
          <p className="masthead-subtitle">House price index estimation</p>
        </div>
        <span className="status-badge">
          <span
            className={`status-dot ${
              online === null ? '' : online ? 'online' : 'offline'
            }`}
          />
          {online === null ? 'checking backend…' : online ? 'backend online' : 'backend offline'}
        </span>
      </header>

      <nav className="tab-nav">
        <button
          className={`tab-button ${tab === 'predict' ? 'active' : ''}`}
          onClick={() => setTab('predict')}
        >
          Predict
        </button>
        <button
          className={`tab-button ${tab === 'history' ? 'active' : ''}`}
          onClick={() => setTab('history')}
        >
          History
        </button>
      </nav>

      {tab === 'predict' ? (
        <div className="grid">
          <PredictionForm onPredict={handlePredict} submitting={submitting} error={error} />
          <IndexCard latest={latest} />
        </div>
      ) : (
        <HistoryTable history={history} onClear={handleClear} />
      )}
    </div>
  )
}
