import fs from "node:fs";
import path from "node:path";
import pptxgen from "/Users/amayasbariz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs/dist/pptxgen.cjs.js";

const baseDir = process.cwd();
const metadata = JSON.parse(fs.readFileSync(path.join(baseDir, "models/model_metadata.json"), "utf8"));
const comparison = JSON.parse(fs.readFileSync(path.join(baseDir, "models/model_comparison.json"), "utf8"));
const shapGlobal = fs.existsSync(path.join(baseDir, "reports/shap_global_importance.csv"))
  ? fs.readFileSync(path.join(baseDir, "reports/shap_global_importance.csv"), "utf8").trim().split("\n").slice(1).map((line) => {
      const [feature, meanAbsShap, meanShap] = line.split(",");
      return { feature, mean_abs_shap: Number(meanAbsShap), mean_shap: Number(meanShap) };
    })
  : [];

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Projet EFREI";
pptx.subject = "Customer churn prediction";
pptx.title = "Système intelligent de rétention client";
pptx.company = "EFREI";
pptx.lang = "fr-FR";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "fr-FR",
};

const C = {
  navy: "17324D",
  blue: "2563EB",
  teal: "0F766E",
  green: "16A34A",
  amber: "D97706",
  red: "DC2626",
  ink: "111827",
  muted: "6B7280",
  line: "D1D5DB",
  pale: "F3F6FA",
  white: "FFFFFF",
};

function title(slide, text, kicker = "") {
  if (kicker) slide.addText(kicker, { x: 0.55, y: 0.28, w: 12.2, h: 0.22, fontSize: 8.5, color: C.teal, bold: true, margin: 0 });
  slide.addText(text, { x: 0.55, y: 0.55, w: 12.2, h: 0.45, fontSize: 21, bold: true, color: C.ink, margin: 0 });
  slide.addShape(pptx.ShapeType.line, { x: 0.55, y: 1.12, w: 12.2, h: 0, line: { color: C.line, width: 1 } });
}

function foot(slide) {
  slide.addText("Projet Data Science M2 | Rétention client", { x: 0.55, y: 7.05, w: 6, h: 0.18, fontSize: 7, color: C.muted, margin: 0 });
}

function metric(slide, x, y, label, value, color = C.blue) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w: 2.55, h: 1.05, rectRadius: 0.05, fill: { color: C.pale }, line: { color: "E5E7EB", width: 1 } });
  slide.addText(value, { x: x + 0.15, y: y + 0.18, w: 2.2, h: 0.36, fontSize: 19, bold: true, color, margin: 0 });
  slide.addText(label, { x: x + 0.15, y: y + 0.64, w: 2.2, h: 0.24, fontSize: 8.5, color: C.muted, margin: 0 });
}

function bullets(slide, items, x, y, w, size = 13) {
  slide.addText(items.map((text) => ({ text, options: { bullet: { indent: 12 }, hanging: 4 } })), {
    x, y, w, h: 3.8, fontSize: size, color: C.ink, breakLine: false, fit: "shrink", valign: "top",
  });
}

let s = pptx.addSlide();
s.background = { color: C.navy };
s.addText("Système intelligent de rétention client", { x: 0.7, y: 1.0, w: 8.7, h: 0.7, fontSize: 30, bold: true, color: C.white, margin: 0 });
s.addText("Prédiction du churn, comparaison multi-modèles, dashboard décisionnel et API optionnelle", { x: 0.72, y: 1.85, w: 8.8, h: 0.45, fontSize: 15, color: "DBEAFE", margin: 0 });
metric(s, 0.72, 3.0, "clients analysés", `${metadata.dataset_rows.toLocaleString("fr-FR")}`, C.white);
metric(s, 3.55, 3.0, "modèle final", metadata.model_type, C.white);
metric(s, 6.38, 3.0, "ROC-AUC test", metadata.roc_auc.toFixed(3), C.white);
s.addText("EFREI - Data Engineering & AI", { x: 0.72, y: 6.65, w: 5, h: 0.25, fontSize: 10, color: "BFDBFE", margin: 0 });

s = pptx.addSlide();
title(s, "Problématique métier", "1 | Cadrage");
bullets(s, [
  "Anticiper les clients susceptibles de résilier leur abonnement.",
  "Prioriser les actions CRM avec un score de risque exploitable.",
  "Comparer ML classique et Deep Learning avec une démarche défendable.",
  "Transformer le modèle en outil métier via Streamlit.",
], 0.8, 1.6, 6.0);
metric(s, 7.4, 1.55, "taux de churn", `${(metadata.positive_class_rate * 100).toFixed(1)}%`, C.red);
metric(s, 10.15, 1.55, "features", `${metadata.n_features}`, C.teal);
metric(s, 7.4, 2.9, "train", `${metadata.train_set_size.toLocaleString("fr-FR")}`, C.blue);
metric(s, 10.15, 2.9, "test", `${metadata.test_set_size.toLocaleString("fr-FR")}`, C.blue);
foot(s);

s = pptx.addSlide();
title(s, "Pipeline data sans leakage", "2 | Méthode");
bullets(s, [
  "Split train/test stratifié pour conserver la proportion de churn.",
  "ColumnTransformer intégré aux pipelines sklearn.",
  "StandardScaler sur les variables numériques.",
  "OneHotEncoder(handle_unknown='ignore') sur les catégories.",
  "customer_id exclu : identifiant technique non généralisable.",
  "Seuil de décision ajusté sur validation pour éviter F1=0.",
], 0.8, 1.45, 7.2, 12.5);
s.addShape(pptx.ShapeType.roundRect, { x: 8.55, y: 1.55, w: 3.9, h: 3.1, fill: { color: "ECFDF5" }, line: { color: "A7F3D0", width: 1 } });
s.addText("Train only", { x: 8.9, y: 1.85, w: 3.2, h: 0.3, fontSize: 18, bold: true, color: C.teal, margin: 0 });
s.addText("Les encodeurs et scalers sont appris uniquement sur les données d'entraînement, puis appliqués au test.", { x: 8.9, y: 2.35, w: 3.0, h: 1.1, fontSize: 13, color: C.ink, fit: "shrink", margin: 0 });
foot(s);

s = pptx.addSlide();
title(s, "Comparaison quantitative des modèles", "3 | Résultats");
const rows = comparison.models.map((m) => [m.model, m.accuracy.toFixed(3), m.precision.toFixed(3), m.recall.toFixed(3), m.f1_score.toFixed(3), m.roc_auc.toFixed(3)]);
s.addTable([["Modèle", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC"], ...rows], {
  x: 0.65, y: 1.45, w: 12, h: 2.5,
  border: { color: C.line, width: 1 },
  fontFace: "Aptos", fontSize: 10,
  color: C.ink,
  fill: { color: C.white },
  margin: 0.06,
});
s.addText(`Modèle retenu : ${metadata.model_type}`, { x: 0.75, y: 4.45, w: 5.5, h: 0.38, fontSize: 18, bold: true, color: C.blue, margin: 0 });
s.addText("Sélection par ROC-AUC, avec seuil métier optimisé pour détecter la classe churn au lieu de maximiser seulement l'accuracy.", { x: 0.75, y: 4.95, w: 9.2, h: 0.7, fontSize: 13, color: C.ink, margin: 0 });
foot(s);

s = pptx.addSlide();
title(s, "Correction du problème F1 = 0", "4 | Deep Learning");
const mlp = comparison.models.find((m) => m.model === "MLP");
metric(s, 0.8, 1.55, "F1 MLP après correction", mlp.f1_score.toFixed(3), C.green);
metric(s, 3.65, 1.55, "Recall MLP", mlp.recall.toFixed(3), C.green);
metric(s, 6.5, 1.55, "ROC-AUC MLP", mlp.roc_auc.toFixed(3), C.green);
bullets(s, [
  "Le F1 nul venait d'un seuil 0.5 trop strict dans un dataset déséquilibré.",
  "Le seuil est maintenant optimisé sur validation.",
  "Le MLP devient comparable, mais reste inférieur aux modèles d'ensemble sur données tabulaires.",
], 0.9, 3.2, 8.8, 13);
foot(s);

s = pptx.addSlide();
title(s, "Interprétabilité SHAP", "5 | Explainability");
const top = (shapGlobal.length ? shapGlobal : metadata.feature_importance.map((d) => ({ feature: d.feature, mean_abs_shap: d.importance }))).slice(0, 8);
const maxImp = Math.max(...top.map((d) => d.mean_abs_shap));
top.forEach((d, i) => {
  const y = 1.45 + i * 0.45;
  s.addText(d.feature, { x: 0.8, y, w: 3.0, h: 0.25, fontSize: 10, color: C.ink, margin: 0 });
  s.addShape(pptx.ShapeType.rect, { x: 3.95, y: y + 0.03, w: 6.2 * (d.mean_abs_shap / maxImp), h: 0.18, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText(d.mean_abs_shap.toFixed(4), { x: 10.35, y, w: 1.0, h: 0.25, fontSize: 9, color: C.muted, margin: 0 });
});
s.addText("Méthode : SHAP sur le modèle final. Les valeurs expliquent l impact des variables sur la classe churn, avec une lecture globale et des exemples locaux.", { x: 0.8, y: 5.45, w: 9.6, h: 0.55, fontSize: 12, color: C.ink, margin: 0 });
foot(s);

s = pptx.addSlide();
title(s, "Dashboard et industrialisation", "6 | MVP");
bullets(s, [
  "Pilotage : KPI, churn par segment, revenu à risque.",
  "Prédiction : formulaire de scénario et vraie inférence locale du modèle final.",
  "Modèles : comparaison des quatre algorithmes et seuils de décision.",
  "Explicabilité : page dédiée aux résultats SHAP globaux et locaux.",
  "API FastAPI disponible comme bonus, mais non nécessaire au dashboard.",
], 0.8, 1.5, 6.8, 13);
s.addShape(pptx.ShapeType.roundRect, { x: 8.25, y: 1.55, w: 3.9, h: 3.6, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
s.addText("Architecture", { x: 8.6, y: 1.9, w: 3.0, h: 0.35, fontSize: 18, bold: true, color: C.blue, margin: 0 });
s.addText("CSV → Pipeline ML → Streamlit\nOption : FastAPI /predict", { x: 8.6, y: 2.55, w: 3.0, h: 1.3, fontSize: 15, color: C.ink, fit: "shrink", margin: 0 });
foot(s);

s = pptx.addSlide();
title(s, "Conclusion et prochaines étapes", "7 | Synthèse");
bullets(s, [
  "Le cahier des charges principal est couvert : data, 4 modèles, DL, dashboard, interprétabilité, rapport.",
  "Le meilleur modèle est sélectionné sur une métrique adaptée au déséquilibre.",
  "Le dashboard est autonome et exploitable par un profil métier.",
  "Extensions : monitoring du drift, déploiement cloud complet, tests d'API, versioning Git propre.",
], 0.8, 1.55, 9.8, 14);
foot(s);

const out = path.join(baseDir, "reports", "presentation_churn_retention.pptx");
await pptx.writeFile({ fileName: out });
console.log(out);
