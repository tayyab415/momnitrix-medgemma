import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MamaGuard Localhost Simulator",
  description: "Controlled simulation environment for physiological vitals and multi-modal health data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
