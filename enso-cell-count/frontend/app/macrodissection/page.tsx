import type { Metadata } from "next";
import MacrodissectionClient from "./MacrodissectionClient";

export const metadata: Metadata = {
  title: "EnsoPurity Macrodissection Workbench",
  description:
    "Pathologist-in-the-loop ROI selection and adequacy estimation for downstream molecular testing.",
};

export default function MacrodissectionPage() {
  return <MacrodissectionClient />;
}
