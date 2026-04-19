import type { Metadata } from "next";
import { Inter, Roboto_Mono } from "next/font/google"; // Bloomberg terminal vibe typefaces
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const robotoMono = Roboto_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Confidence Sentinel | AI Reliability Monitoring",
  description: "Real-time AI reliability monitoring platform.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`dark ${inter.variable} ${robotoMono.variable} h-full antialiased`}
      style={{ backgroundColor: "#09090b", color: "#ededed" }}
    >
      <body className="min-h-full flex flex-col font-sans bg-[#09090b] text-[#ededed] overflow-hidden">
        {children}
      </body>
    </html>
  );
}
