export default function HistoryTable({ history, onClear }) {
  return (
    <div className="panel">
      <div className="history-toolbar">
        <p className="panel-label" style={{ margin: 0 }}>
          Ledger · Past Estimates
        </p>
        {history.length > 0 && (
          <button className="clear-button" onClick={onClear}>
            Clear history
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <p className="ledger-empty">No estimates recorded yet.</p>
      ) : (
        <table className="ledger">
          <thead>
            <tr>
              <th>Time</th>
              <th>Country</th>
              <th>Year</th>
              <th>HICP</th>
              <th>Unempl.</th>
              <th>HPI</th>
            </tr>
          </thead>
          <tbody>
            {history.map((entry) => (
              <tr key={entry.id}>
                <td>{new Date(entry.timestamp).toLocaleTimeString()}</td>
                <td>{entry.country}</td>
                <td>{entry.year}</td>
                <td>{entry.hicp}</td>
                <td>{entry.unemployment_rate}</td>
                <td>{entry.hpi.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
