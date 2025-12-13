
'use client';

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, RefreshCw, Zap, TrendingUp, TrendingDown, Target, Shield, Newspaper } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function AssetPage({ params }: { params: Promise<{ symbol: string }> }) {
    const { symbol } = React.use(params);
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [timeframe, setTimeframe] = useState("1 Hour");

    const fetchData = async () => {
        setLoading(true);
        try {
            // Reuse the all endpoint and filter (simple for now)
            // In prod, use specific endpoint
            const res = await axios.get('/api/predict_all?assets=' + symbol);
            if (res.data.assets && res.data.assets[symbol]) {
                setData(res.data.assets[symbol]);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, [symbol]);

    if (loading) return <div className="min-h-screen bg-background text-white p-8 animate-pulse flex items-center justify-center">Loading Asset Intelligence...</div>;
    if (!data) return <div className="min-h-screen bg-background text-danger p-8">Asset Not Found</div>;

    const decision = data.decision;
    const mainSignal = decision.decision;
    const activePred = data.predictions[timeframe] || {};
    const smc = data.smc || {};
    const micro = data.microstructure || {};

    return (
        <div className="min-h-screen bg-background text-white p-6 md:p-8 space-y-8">
            {/* HEADER */}
            <div className="flex items-center justify-between border-b border-border pb-6">
                <div className="flex items-center gap-4">
                    <Link href="/" className="p-2 hover:bg-secondary rounded-full transition-colors">
                        <ArrowLeft className="w-6 h-6" />
                    </Link>
                    <div>
                        <h1 className="text-4xl font-bold flex items-center gap-3">
                            {symbol}
                            <Badge variant="outline" className="text-lg py-1">{mainSignal}</Badge>
                        </h1>
                        <p className="text-muted-foreground mt-1 flex items-center gap-2">
                            <Shield className="w-4 h-4 text-primary" /> Confidence: {(decision.net_confidence * 100).toFixed(1)}%
                        </p>
                    </div>
                </div>
                <button onClick={fetchData} className="px-4 py-2 bg-secondary rounded hover:bg-secondary/80 flex items-center gap-2">
                    <RefreshCw className="w-4 h-4" /> Refresh
                </button>
            </div>

            {/* TIMEFRAME SELECTOR */}
            <div className="flex gap-2">
                {['Daily', '4 Hour', '1 Hour', '15 Min'].map(tf => (
                    <button
                        key={tf}
                        onClick={() => setTimeframe(tf)}
                        className={cn(
                            "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                            timeframe === tf ? "bg-primary text-black shadow-lg shadow-primary/20" : "bg-card hover:bg-secondary text-muted-foreground"
                        )}
                    >
                        {tf}
                    </button>
                ))}
            </div>

            {/* MAIN GRID */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* 1. SIGNAL CARD */}
                <Card className="lg:col-span-2 border-primary/20 bg-card/50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Zap className="w-5 h-5 text-yellow-500" />
                            AI Prediction ({timeframe})
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-6">
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Signal</p>
                            <p className={cn("text-2xl font-bold", activePred.signal?.includes("Buy") ? "text-success" : activePred.signal?.includes("Sell") ? "text-danger" : "text-gray-400")}>
                                {activePred.signal || "N/A"}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Confidence</p>
                            <p className="text-2xl font-mono text-primary">{(activePred.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Take Profit</p>
                            <p className="text-2xl font-mono text-success">{activePred.levels?.tp || "-"}</p>
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Stop Loss</p>
                            <p className="text-2xl font-mono text-danger">{activePred.levels?.sl || "-"}</p>
                        </div>

                        {/* Technicals */}
                        {activePred.technicals && (
                            <div className="col-span-2 md:col-span-4 mt-4 pt-4 border-t border-border grid grid-cols-3 gap-4">
                                <div>
                                    <p className="text-[10px] text-muted-foreground">RSI (14)</p>
                                    <p className="text-lg font-mono">{activePred.technicals.rsi?.toFixed(1) || "0.0"}</p>
                                </div>
                                <div>
                                    <p className="text-[10px] text-muted-foreground">ADX Trend</p>
                                    <p className="text-lg font-mono">{activePred.technicals.adx?.toFixed(1) || "0.0"}</p>
                                </div>
                                <div>
                                    <p className="text-[10px] text-muted-foreground">EMA Trend</p>
                                    <p className="text-lg font-mono">{activePred.technicals.close > activePred.technicals.ema50 ? "BULLISH" : "BEARISH"}</p>
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* 2. NEWS CARD */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Newspaper className="w-5 h-5 text-blue-400" />
                            Market Events
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {data.news && data.news.length > 0 ? (
                            <div className="space-y-4">
                                {data.news.slice(0, 5).map((n: any, i: number) => (
                                    <div key={i} className="text-sm border-l-2 border-primary/50 pl-3">
                                        <div className="flex justify-between text-xs text-muted-foreground">
                                            <span>{n.currency}</span>
                                            <span>{n.impact}</span>
                                        </div>
                                        <p className="font-medium truncate">{n.event}</p>
                                        <div className="flex gap-2 text-[10px] mt-1 font-mono text-muted-foreground">
                                            <span>Act: {n.actual}</span>
                                            <span>Fcst: {n.forecast}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-sm text-muted-foreground">No high-impact news detected.</p>
                        )}
                    </CardContent>
                </Card>

                {/* 3. SMC PANEL */}
                <Card className="lg:col-span-3">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Target className="w-5 h-5 text-purple-400" />
                            Smart Money Concepts (SMC)
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="grid md:grid-cols-2 gap-8">
                        {/* Order Blocks */}
                        <div>
                            <h4 className="text-sm font-semibold mb-3 text-muted-foreground uppercase">Order Blocks (Zones)</h4>
                            <div className="space-y-2">
                                {smc.bear_obs_found?.map((ob: any, i: number) => (
                                    <div key={`bear-${i}`} className="bg-danger/10 border border-danger/20 p-2 rounded flex justify-between text-xs">
                                        <span className="text-danger font-bold">SUPPLY</span>
                                        <span className="font-mono">{ob.bottom} - {ob.top}</span>
                                    </div>
                                ))}
                                {smc.bull_obs_found?.map((ob: any, i: number) => (
                                    <div key={`bull-${i}`} className="bg-success/10 border border-success/20 p-2 rounded flex justify-between text-xs">
                                        <span className="text-success font-bold">DEMAND</span>
                                        <span className="font-mono">{ob.bottom} - {ob.top}</span>
                                    </div>
                                ))}
                                {(!smc.bear_obs_found && !smc.bull_obs_found) && <p className="text-sm text-muted-foreground">No zones localized.</p>}
                            </div>
                        </div>

                        {/* Microstructure */}
                        <div>
                            <h4 className="text-sm font-semibold mb-3 text-muted-foreground uppercase">Microstructure</h4>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 bg-secondary rounded-lg">
                                    <p className="text-xs text-muted-foreground">Displacement</p>
                                    <p className={cn("text-lg font-bold", micro.displacement ? "text-primary" : "text-muted-foreground")}>
                                        {micro.displacement ? "DETECTED" : "NONE"}
                                    </p>
                                </div>
                                <div className="p-3 bg-secondary rounded-lg">
                                    <p className="text-xs text-muted-foreground">Flow Direction</p>
                                    <p className={cn("text-lg font-bold", micro.displacement_dir === 1 ? "text-success" : micro.displacement_dir === -1 ? "text-danger" : "text-muted-foreground")}>
                                        {micro.displacement_dir === 1 ? "BULLISH" : micro.displacement_dir === -1 ? "BEARISH" : "MIXED"}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
