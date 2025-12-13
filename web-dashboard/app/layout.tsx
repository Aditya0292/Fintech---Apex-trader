
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
    title: "APEX TRADE AI | Institutional Intelligence",
    description: "Multi-Asset Smart Money Concepts & AI Prediction Engine",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="dark">
            <body className={cn(inter.variable, "min-h-screen bg-background font-sans antialiased text-white selection:bg-primary/20")}>
                {children}
            </body>
        </html>
    );
}
