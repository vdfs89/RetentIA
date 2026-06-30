import { useState } from "react";

const API = "http://retentia.vitorsilva.engineer";

const DEFAULTS = {
  gender: "Female", SeniorCitizen: 0, Partner: "Yes", Dependents: "No",
  tenure: 12, PhoneService: "Yes", MultipleLines: "No",
  InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "Yes",
  DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes",
  StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check", MonthlyCharges: 70.35, TotalCharges: 845.5,
};

const FIELDS = [
  { group: "Perfil", fields: [
    { key: "gender", label: "Gênero", type: "select", options: ["Female", "Male"], labels: ["Feminino", "Masculino"] },
    { key: "SeniorCitizen", label: "Idoso", type: "select", options: [0, 1], labels: ["Não", "Sim"] },
    { key: "Partner", label: "Parceiro(a)", type: "select", options: ["Yes", "No"], labels: ["Sim", "Não"] },
    { key: "Dependents", label: "Dependentes", type: "select", options: ["Yes", "No"], labels: ["Sim", "Não"] },
  ]},
  { group: "Contrato", fields: [
    { key: "Contract", label: "Tipo de Contrato", type: "select", options: ["Month-to-month", "One year", "Two year"], labels: ["Mês a mês", "Um ano", "Dois anos"] },
    { key: "PaperlessBilling", label: "Fatura Digital", type: "select", options: ["Yes", "No"], labels: ["Sim", "Não"] },
    { key: "PaymentMethod", label: "Pagamento", type: "select", options: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], labels: ["Cheque eletrônico", "Cheque pelos correios", "Transferência bancária (automática)", "Cartão de crédito (automático)"] },
    { key: "tenure", label: "Tempo de Contrato (meses)", type: "number", min: 0, max: 100 },
  ]},
  { group: "Serviços", fields: [
    { key: "PhoneService", label: "Telefone", type: "select", options: ["Yes", "No"], labels: ["Sim", "Não"] },
    { key: "MultipleLines", label: "Múltiplas Linhas", type: "select", options: ["No phone service", "No", "Yes"], labels: ["Sem telefone", "Não", "Sim"] },
    { key: "InternetService", label: "Internet", type: "select", options: ["DSL", "Fiber optic", "No"], labels: ["DSL", "Fibra Óptica", "Não"] },
    { key: "OnlineSecurity", label: "Segurança Online", type: "select", options: ["No internet service", "No", "Yes"], labels: ["Sem internet", "Não", "Sim"] },
    { key: "OnlineBackup", label: "Backup Online", type: "select", options: ["No internet service", "No", "Yes"], labels: ["Sem internet", "Não", "Sim"] },
    { key: "DeviceProtection", label: "Proteção de Dispositivo", type: "select", options: ["No internet service", "No", "Yes"], labels: ["Sem internet", "Não", "Sim"] },
    { key: "TechSupport", label: "Suporte Técnico", type: "select", options: ["No internet service", "No", "Yes"], labels: ["Sem internet", "Não", "Sim"] },
    { key: "StreamingTV", label: "Streaming TV", type: "select", options: ["No internet service", "No", "Yes"], labels: ["Sem internet", "Não", "Sim"] },
    { key: "StreamingMovies", label: "Streaming Filmes", type: "select", options: ["No internet service", "No", "Yes"], labels: ["Sem internet", "Não", "Sim"] },
  ]},
  { group: "Cobrança", fields: [
    { key: "MonthlyCharges", label: "Cobrança Mensal (R$)", type: "number", step: 0.01, min: 0 },
    { key: "TotalCharges", label: "Cobrança Total (R$)", type: "number", step: 0.01, min: 0 },
  ]},
];

export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const predict = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, SeniorCitizen: Number(form.SeniorCitizen) }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const prob = result ? Math.round(result.churn_probability * 100) : null;
  const isChurn = result?.churn_prediction;

  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", color: "#f8fafc", fontFamily: "system-ui, sans-serif" }}>
      <header style={{ borderBottom: "1px solid #1e293b", padding: "20px 32px", display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 36, height: 36, borderRadius: 8, background: "linear-gradient(135deg, #06b6d4, #7c3aed)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 18, fontWeight: 700 }}>R</div>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.5px" }}>RetentIA</div>
          <div style={{ fontSize: 12, color: "#64748b" }}>Predição de Churn em Tempo Real</div>
        </div>
      </header>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px", display: "grid", gridTemplateColumns: result ? "1fr 300px" : "1fr", gap: 24 }}>
        <div>
          {FIELDS.map(({ group, fields }) => (
            <div key={group} style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", color: "#06b6d4", textTransform: "uppercase", marginBottom: 12 }}>{group}</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))", gap: 12 }}>
                {fields.map(f => (
                  <div key={f.key}>
                    <label style={{ display: "block", fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>{f.label}</label>
                    {f.type === "select" ? (
                      <select value={form[f.key]}
                        onChange={e => set(f.key, f.key === "SeniorCitizen" ? Number(e.target.value) : e.target.value)}
                        style={{ width: "100%", background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: "8px 10px", color: "#f8fafc", fontSize: 13 }}>
                        {f.options.map((opt, i) => <option key={opt} value={opt}>{f.labels ? f.labels[i] : opt}</option>)}
                      </select>
                    ) : (
                      <input type="number" value={form[f.key]} min={f.min} max={f.max} step={f.step || 1}
                        onChange={e => set(f.key, Number(e.target.value))}
                        style={{ width: "100%", background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: "8px 10px", color: "#f8fafc", fontSize: 13, boxSizing: "border-box" }} />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}

          <button onClick={predict} disabled={loading}
            style={{ width: "100%", padding: 14, borderRadius: 8, border: "none", background: loading ? "#334155" : "linear-gradient(135deg, #06b6d4, #7c3aed)", color: "#fff", fontSize: 15, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer" }}>
            {loading ? "Analisando..." : "Analisar Risco de Churn"}
          </button>

          {error && (
            <div style={{ marginTop: 12, padding: "12px 16px", background: "#450a0a", border: "1px solid #7f1d1d", borderRadius: 8, color: "#fca5a5", fontSize: 13 }}>
              {error.includes("fetch") ? "Não foi possível conectar à API (CORS ou offline)." : `Erro: ${error}`}
            </div>
          )}
        </div>

        {result && (
          <div style={{ position: "sticky", top: 24, alignSelf: "start" }}>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: 24, border: `1px solid ${isChurn ? "#7c3aed" : "#0891b2"}` }}>
              <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 16 }}>Resultado</div>
              <div style={{ textAlign: "center", marginBottom: 20 }}>
                <svg viewBox="0 0 120 70" style={{ width: "100%" }}>
                  <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="#1e3a5f" strokeWidth="12" strokeLinecap="round"/>
                  <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none"
                    stroke={prob > 70 ? "#ef4444" : prob > 40 ? "#f59e0b" : "#22c55e"}
                    strokeWidth="12" strokeLinecap="round"
                    strokeDasharray={`${prob * 1.57} 157`}/>
                  <text x="60" y="55" textAnchor="middle" fill="#f8fafc" fontSize="20" fontWeight="700">{prob}%</text>
                </svg>
                <div style={{ fontSize: 13, color: "#94a3b8", marginTop: -8 }}>probabilidade de churn</div>
              </div>
              <div style={{ padding: "12px 16px", borderRadius: 8, background: isChurn ? "#3b0764" : "#052e16", border: `1px solid ${isChurn ? "#7c3aed" : "#166534"}`, textAlign: "center", marginBottom: 16 }}>
                <div style={{ fontSize: 22 }}>{isChurn ? "⚠️" : "✅"}</div>
                <div style={{ fontWeight: 700, fontSize: 15, color: isChurn ? "#c4b5fd" : "#86efac", marginTop: 4 }}>
                  {isChurn ? "Alto Risco de Churn" : "Baixo Risco"}
                </div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
                  {isChurn ? "Recomenda-se ação de retenção" : "Cliente provavelmente permanece"}
                </div>
              </div>
              {[["Probabilidade", `${prob}%`], ["Limite de Decisão", "36% (custo-ótimo)"], ["Decisão", isChurn ? "Churn" : "Retido"]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
                  <span style={{ color: "#64748b" }}>{k}</span>
                  <span style={{ color: "#f8fafc", fontWeight: 500 }}>{v}</span>
                </div>
              ))}
              <div style={{ marginTop: 14, padding: "10px 12px", background: "#0f172a", borderRadius: 6, fontSize: 11, color: "#475569" }}>
                C_FN=500 › C_FP=100 → threshold 0.36
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
