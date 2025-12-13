
'use client';

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, TrendingUp, TrendingDown, AlertTriangle, ShieldCheck, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types (simplified)
interface AssetData {
    symbol: string;
    predictions: Record<string, any>;
    decision: { decision: string; net_confidence: any; reason: string };
    smc?: any;
    microstructure?: any;
}
interface DashboardData {
    generated_at: string;
    market_context: { csm: Record<string, number> };
    assets: Record<string, AssetData>;
    ranking: any[];
}

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const fetchData = async (forceRefresh = false) => {
        setLoading(true);
        try {
            // Call our Next.js API route
            // Force refresh bypasses the 5m cache
            const url = `/api/predict_all?assets=all${forceRefresh ? '&refresh=true' : ''}`;
            const res = await axios.get(url);
            if (res.data.error) throw new Error(res.data.error);
            setData(res.data);
            setError('');
        } catch (err: any) {
            setError(err.message || 'Failed to fetch data');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData(); // Default: Use Cache
    }, []);

    // --- RENDERING HELPERS ---

    const getSignalBadge = (signal: string) => {
        const s = signal.toUpperCase();
        if (s.includes('BUY')) return <Badge variant="success">BUY</Badge>;
        if (s.includes('SELL')) return <Badge variant="danger">SELL</Badge>;
        return <Badge variant="secondary">WAIT</Badge>;
    };

    if (loading && !data) {
        return (
            <div className="min-h-screen bg-background p-6 md:p-8 space-y-8 text-white">
                {/* Skeleton Header */}
                <div className="flex justify-between items-center border-b border-border pb-6">
                    <div className="space-y-2">
                        <div className="h-8 w-48 bg-secondary/50 rounded animate-pulse" />
                        <div className="h-4 w-64 bg-secondary/30 rounded animate-pulse" />
                    </div>
                </div>

                {/* Skeleton Stats Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Ranking Skeleton */}
                    <div className="lg:col-span-2 h-[300px] bg-card rounded-xl border border-border p-6 space-y-4 animate-pulse">
                        <div className="h-6 w-32 bg-secondary/50 rounded" />
                        <div className="space-y-3">
                            {[1, 2, 3, 4].map(i => (
                                <div key={i} className="h-10 w-full bg-secondary/20 rounded" />
                            ))}
                        </div>
                    </div>
                    {/* CSM Skeleton */}
                    <div className="h-[300px] bg-card rounded-xl border border-border p-6 space-y-4 animate-pulse">
                        <div className="h-6 w-32 bg-secondary/50 rounded" />
                        <div className="space-y-3">
                            {[1, 2, 3, 4, 5, 6].map(i => (
                                <div key={i} className="flex gap-2 items-center">
                                    <div className="h-4 w-8 bg-secondary/30 rounded" />
                                    <div className="flex-1 h-2 bg-secondary/30 rounded" />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Skeleton Asset Grid */}
                <div>
                    <div className="h-6 w-32 bg-secondary/50 rounded mb-6 animate-pulse" />
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                        {[1, 2, 3, 4].map(i => (
                            <div key={i} className="h-[200px] bg-card rounded-xl border border-border p-6 space-y-4 animate-pulse">
                                <div className="flex justify-between">
                                    <div className="h-6 w-24 bg-secondary/50 rounded" />
                                    <div className="h-6 w-12 bg-secondary/50 rounded" />
                                </div>
                                <div className="h-20 bg-secondary/20 rounded" />
                                <div className="flex justify-between pt-4 border-t border-border/50">
                                    <div className="h-4 w-16 bg-secondary/30 rounded" />
                                    <div className="h-4 w-24 bg-secondary/30 rounded" />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen bg-background text-danger">
                <AlertTriangle className="h-16 w-16 mb-4" />
                <h1 className="text-2xl font-bold">SYSTEM ERROR</h1>
                <p>{error}</p>
                <button onClick={() => fetchData(true)} className="mt-4 px-4 py-2 bg-secondary rounded hover:bg-secondary/80">Retry</button>
            </div>
        );
    }

    if (!data) return null;

    // Sorting Ranking
    const ranking = data.ranking || [];
    const topPick = ranking.length > 0 ? ranking[0] : null;

    return (
        <div className="min-h-screen bg-background p-6 md:p-8 space-y-8 text-white">
            {/* HEADER */}
            <header className="flex justify-between items-center border-b border-border pb-6">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                        APEX TRADE AI
                    </h1>
                    <p className="text-muted-foreground text-sm uppercase tracking-widest mt-1">
                        Institutional Multi-Asset Intelligence
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <Badge variant="outline" className="py-1 px-3 border-primary/30 text-primary">
                        <ShieldCheck className="w-3 h-3 mr-2" /> REGULATORY CHECK PASS
                    </Badge>
                    <button onClick={() => fetchData(true)} disabled={loading} className="p-2 rounded-full hover:bg-secondary transition-colors disabled:opacity-50">
                        <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* ⚠️ DISCLAIMER BANNER */}
            <div className="bg-yellow-500/5 border border-yellow-500/20 p-3 rounded-lg flex items-center gap-3 text-xs text-yellow-500/80">
                <AlertTriangle className="w-4 h-4" />
                <span>RISK WARNING: AI predictions are probabilistic. Trading involves checking regulatory compliance in your jurisdiction. Not financial advice.</span>
            </div>

            {/* TOP ROW: RANKING & CSM */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Opportunity Runway (Ranking) */}
                <Card className="lg:col-span-2 border-primary/10">
                    <CardHeader>
                        <CardTitle className="flex items-center">
                            <TrendingUp className="w-5 h-5 mr-2 text-primary" /> Opportunity Runway
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left">
                                <thead className="text-muted-foreground border-b border-border">
                                    <tr>
                                        <th className="py-2 pl-2">Asset</th>
                                        <th>Action</th>
                                        <th>Conf</th>
                                        <th>Score</th>
                                        <th>Rating</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {ranking.map((item: any, idx: number) => (
                                        <motion.tr
                                            key={item.symbol}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: idx * 0.1 }}
                                            className="border-b border-border/50 hover:bg-secondary/30 transition-colors"
                                        >
                                            <td className="py-4 pl-2 font-medium flex items-center">
                                                {idx === 0 && <Zap className="w-3 h-3 text-yellow-500 mr-2" />}
                                                {item.symbol}
                                            </td>
                                            <td>{getSignalBadge(item.direction)}</td>
                                            <td className="font-mono text-primary">{typeof item.confidence === 'string' ? item.confidence : (item.confidence * 100).toFixed(1) + '%'}</td>
                                            <td>{item.score.toFixed(1)}</td>
                                            <td className="text-xs tracking-tighter text-muted-foreground">
                                                {"█".repeat(Math.min(10, Math.floor(item.score / 10)))}
                                                <span className="opacity-20">{"█".repeat(Math.max(0, 10 - Math.floor(item.score / 10)))}</span>
                                            </td>
                                        </motion.tr>
                                    ))}
                                    {ranking.length === 0 && <tr><td colSpan={5} className="py-4 text-center text-muted-foreground">No high-probability setups active.</td></tr>}
                                </tbody>
                            </table>
                        </div>
                    </CardContent>
                </Card>

                {/* CSM */}
                <Card>
                    <CardHeader>
                        <CardTitle>Currency Strength</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {Object.entries(data.market_context.csm)
                                .sort(([, a], [, b]) => b - a)
                                .map(([curr, val]) => (
                                    <div key={curr} className="flex items-center gap-3">
                                        <span className="w-8 font-bold text-sm">{curr}</span>
                                        <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-blue-500 to-primary transition-all duration-1000"
                                                style={{ width: `${(val / 10) * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-xs font-mono w-8 text-right">{val.toFixed(1)}</span>
                                    </div>
                                ))}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* HEATMAP GRID */}
            <h2 className="text-xl font-bold mt-8">Active Assets</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                {Object.values(data.assets).map((asset) => {
                    // Extract main signal (e.g., from 4h or decision)
                    const dec = asset.decision;
                    const isBullish = dec.decision.includes("BUY");
                    const isBearish = dec.decision.includes("SELL");
                    const signalColor = isBullish ? "border-success/50" : isBearish ? "border-danger/50" : "border-border";

                    return (
                        <Link href={`/asset/${asset.symbol}`} key={asset.symbol} className="block group">
                            <Card className={`group-hover:-translate-y-1 transition-all duration-300 ${signalColor} hover:shadow-lg hover:shadow-primary/5 h-full`}>
                                <CardHeader className="flex flex-row items-center justify-between pb-2">
                                    <CardTitle>{asset.symbol}</CardTitle>
                                    {getSignalBadge(dec.decision)}
                                </CardHeader>
                                <CardContent>
                                    <div className="text-sm text-muted-foreground mb-4 font-mono">
                                        Confidence: <span className="text-foreground font-bold">{(dec.net_confidence * 100).toFixed(1)}%</span>
                                    </div>

                                    {/* Timeframe Grid */}
                                    <div className="grid grid-cols-4 gap-1 mb-4">
                                        {['Daily', '4 Hour', '1 Hour', '15 Min'].map(tf => {
                                            const p = asset.predictions[tf];
                                            if (!p) return <div key={tf} className="h-1 bg-secondary rounded"></div>;
                                            const col = p.signal.includes("Buy") ? "bg-success" : p.signal.includes("Sell") ? "bg-danger" : "bg-secondary";
                                            return (
                                                <div key={tf} className="flex flex-col items-center gap-1 group/tf relative">
                                                    <div className={`w-full h-1.5 rounded-full ${col} transition-all`}></div>
                                                    <span className="text-[10px] text-muted-foreground">{tf.replace(" Hour", "H").replace(" Min", "m")}</span>
                                                </div>
                                            )
                                        })}
                                    </div>

                                    <div className="space-y-2 text-xs border-t border-border pt-3">
                                        <div className="flex justify-between">
                                            <span>Reason</span>
                                            <span className="text-right text-muted-foreground truncate w-32">{dec.reason}</span>
                                        </div>
                                        {asset.smc && (
                                            <div className="flex justify-between">
                                                <span>Structure</span>
                                                <span className="font-mono text-primary">SMC Active</span>
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        </Link>
                    );
                })}
            </div>

            <footer className="text-center text-xs text-muted-foreground mt-12 pb-6">
                System Latency: 12ms • Data Freshness: {new Date(data.generated_at).toLocaleTimeString()} • Build v2.4 (Institutional)
            </footer>
        </div>
    );
}
