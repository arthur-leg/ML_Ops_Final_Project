import { useState } from 'react'
import { COUNTRIES } from '../countries.js'

const initialForm = {
  country: 'AT',
  year: new Date().getFullYear(),
  hicp: '',
  unemployment_rate: '',
}

export default function PredictionForm({ onPredict, submitting, error }) {
  const [form, setForm] = useState(initialForm)

  function handleChange(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onPredict({
      country: form.country,
      year: Number(form.year),
      hicp: Number(form.hicp),
      unemployment_rate: Number(form.unemployment_rate),
    })
  }

  return (
    <form className="panel" onSubmit={handleSubmit}>
      <p className="panel-label">Input · New Estimate</p>

      {error && <div className="form-error">{error}</div>}

      <div className="field">
        <label htmlFor="country">Country</label>
        <select
          id="country"
          value={form.country}
          onChange={(e) => handleChange('country', e.target.value)}
        >
          {COUNTRIES.map((c) => (
            <option key={c.code} value={c.code}>
              {c.name} ({c.code})
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="year">Year</label>
        <input
          id="year"
          type="number"
          min="2000"
          max="2035"
          value={form.year}
          onChange={(e) => handleChange('year', e.target.value)}
          required
        />
      </div>

      <div className="field">
        <label htmlFor="hicp">HICP (inflation rate, %)</label>
        <input
          id="hicp"
          type="number"
          step="0.1"
          placeholder="e.g. 1.2"
          value={form.hicp}
          onChange={(e) => handleChange('hicp', e.target.value)}
          required
        />
      </div>

      <div className="field">
        <label htmlFor="unemployment_rate">Unemployment rate (%)</label>
        <input
          id="unemployment_rate"
          type="number"
          step="0.1"
          placeholder="e.g. 5.4"
          value={form.unemployment_rate}
          onChange={(e) => handleChange('unemployment_rate', e.target.value)}
          required
        />
      </div>

      <button className="submit-button" type="submit" disabled={submitting}>
        {submitting ? 'Estimating…' : 'Estimate HPI'}
      </button>
    </form>
  )
}
