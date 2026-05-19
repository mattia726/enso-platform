/**
 * TCGA project_id → human-readable cancer type (for investors/doctors).
 * Source: GDC TCGA Study Abbreviations.
 */
export const TCGA_DISPLAY_NAMES: Record<string, string> = {
  "TCGA-ACC": "Adrenocortical Carcinoma",
  "TCGA-BLCA": "Bladder Cancer",
  "TCGA-BRCA": "Breast Cancer",
  "TCGA-CESC": "Cervical Cancer",
  "TCGA-CHOL": "Cholangiocarcinoma",
  "TCGA-COAD": "Colon Adenocarcinoma",
  "TCGA-DLBC": "Lymphoid Neoplasm",
  "TCGA-ESCA": "Esophageal Carcinoma",
  "TCGA-GBM": "Glioblastoma",
  "TCGA-HNSC": "Head and Neck Cancer",
  "TCGA-KICH": "Kidney Chromophobe",
  "TCGA-KIRC": "Kidney Clear Cell Carcinoma",
  "TCGA-KIRP": "Kidney Papillary Cell Carcinoma",
  "TCGA-LGG": "Lower Grade Glioma",
  "TCGA-LIHC": "Liver Cancer",
  "TCGA-LUAD": "Lung Adenocarcinoma",
  "TCGA-LUSC": "Lung Squamous Cell Carcinoma",
  "TCGA-MESO": "Mesothelioma",
  "TCGA-OV": "Ovarian Cancer",
  "TCGA-PAAD": "Pancreatic Cancer",
  "TCGA-PCPG": "Pheochromocytoma",
  "TCGA-PRAD": "Prostate Cancer",
  "TCGA-READ": "Rectal Adenocarcinoma",
  "TCGA-SARC": "Sarcoma",
  "TCGA-SKCM": "Skin Cutaneous Melanoma",
  "TCGA-STAD": "Stomach Adenocarcinoma",
  "TCGA-TGCT": "Testicular Germ Cell",
  "TCGA-THCA": "Thyroid Carcinoma",
  "TCGA-THYM": "Thymoma",
  "TCGA-UCEC": "Uterine Corpus Endometrial Carcinoma",
  "TCGA-UCS": "Uterine Carcinosarcoma",
  "TCGA-UVM": "Uveal Melanoma",
};

export function getCancerDisplayName(projectId: string): string {
  return TCGA_DISPLAY_NAMES[projectId] ?? projectId.replace("TCGA-", "");
}
