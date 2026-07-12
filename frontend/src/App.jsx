import { checkHealth, predictHpi, loginWithGoogle } from './api.js'
import { useEffect, useState } from 'react'
import { GoogleLogin, googleLogout } from '@react-oauth/google'
import PredictionForm from './components/PredictionForm.jsx'
import IndexCard from './components/IndexCard.jsx'
import HistoryTable from './components/HistoryTable.jsx'

const STORAGE_KEY = 'hpi-prediction-history'
const JWT_KEY = 'hpi-jwt'
const USER_KEY = 'hpi-user'

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
  const [jwt, setJwt] = useState(() => localStorage.getItem(JWT_KEY))
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  })

  useEffect(() => {
    checkHealth().then(setOnline)
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
  }, [history])

async function handleLoginSuccess(credentialResponse) {
  setError(null)
  try {
    const data = await loginWithGoogle(credentialResponse.credential)
    localStorage.setItem(JWT_KEY, data.access_token)
    localStorage.setItem(USER_KEY, JSON.stringify({ email: data.email, name: data.name }))
    setJwt(data.access_token)
    setUser({ email: data.email, name: data.name })
  } catch (err) {
    setError('Impossible de se connecter avec Google.')
  }
}

  function handleLogout() {
    googleLogout()
    localStorage.removeItem(JWT_KEY)
    localStorage.removeItem(USER_KEY)
    setJwt(null)
    setUser(null)
  }

  async function handlePredict(payload) {
    setSubmitting(true)
    setError(null)
    try {
      const result = await predictHpi(payload, jwt)
      const entry = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        ...payload,
        hpi: result.hpi,
      }
      setHistory((prev) => [entry, ...prev])
    } catch (err) {
      if (err.status === 401) {
        handleLogout()
        setError('Session expirée, reconnecte-toi.')
      } else {
        setError(err.message)
      }
    } finally {
      setSubmitting(false)
    }
  }

  function handleClear() {
    setHistory([])
  }

  const latest = history[0] ?? null

  if (!jwt) {
    return (
      <div className="app-shell">
        <header className="masthead">
          <div>
            <h1 className="masthead-title">HPI Index Bulletin</h1>
            <p className="masthead-subtitle">House price index estimation</p>
          </div>
        </header>
        <div className="login-screen">
          <p>Connecte-toi pour accéder au bulletin d'indice.</p>
          <GoogleLogin
            onSuccess={handleLoginSuccess}
            onError={() => setError('La connexion Google a échoué.')}
          />
          {error && <p className="error-text">{error}</p>}
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <header className="masthead">
        <div>
          <h1 className="masthead-title">HPI Index Bulletin</h1>
          <p className="masthead-subtitle">House price index estimation</p>
        </div>
        <div className="header-right">
          <span className="status-badge">
            <span
              className={`status-dot ${
                online === null ? '' : online ? 'online' : 'offline'
              }`}
            />
            {online === null ? 'checking backend…' : online ? 'backend online' : 'backend offline'}
          </span>
          {user && (
            <span className="user-badge">
              {user.name ?? user.email}
              <button className="logout-button" onClick={handleLogout}>Se déconnecter</button>
            </span>
          )}
        </div>
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