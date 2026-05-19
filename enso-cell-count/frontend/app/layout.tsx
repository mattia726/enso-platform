import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";

export const metadata: Metadata = {
  title: "Enso Biosciences | Tumor Purity Prediction",
  description:
    "EnsoPurity: AI-powered tumor purity from H&E whole-slide images. 2× more accurate than human pathologists.",
  icons: {
    icon: "/enso-logo.png",
    apple: "/enso-logo.png",
  },
};

export default async function RootLayout({
  children,
  params,
  searchParams,
}: Readonly<{
  children: React.ReactNode;
  params?: Promise<Record<string, string | string[]>>;
  searchParams?: Promise<Record<string, string | string[]>>;
}>) {
  await Promise.all([
    params ?? Promise.resolve({}),
    searchParams ?? Promise.resolve({}),
  ]);
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
