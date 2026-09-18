const PRODUCTION_API = "https://qcas-df-api.onrender.com";

window.QCAS_API_BASE =
  (location.hostname === "localhost" || location.hostname === "127.0.0.1")
    ? "http://localhost:8000"
    : PRODUCTION_API;
