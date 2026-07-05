const BASE_INDEX = 100

export default function IndexCard({ latest }) {
  return (
    <div className="panel index-card">
      <p className="panel-label">Result · Predicted Index</p>

      {!latest ? (
        <p className="index-empty">
          Submit an estimate to see the predicted house price index here.
          Index is relative to a base value of {BASE_INDEX}.
        </p>
      ) : (
        <>
          <div className="index-value-row">
            <span className="index-value">{latest.hpi.toFixed(1)}</span>
            <span className={`index-delta ${latest.hpi >= BASE_INDEX ? 'positive' : 'negative'}`}>
              {latest.hpi >= BASE_INDEX ? '▲' : '▼'} {Math.abs(latest.hpi - BASE_INDEX).toFixed(1)} vs base
            </span>
          </div>

          <dl className="index-meta">
            <dt>country</dt>
            <dd>{latest.country}</dd>
            <dt>year</dt>
            <dd>{latest.year}</dd>
            <dt>hicp</dt>
            <dd>{latest.hicp}%</dd>
            <dt>unemployment</dt>
            <dd>{latest.unemployment_rate}%</dd>
          </dl>
        </>
      )}
    </div>
  )
}
