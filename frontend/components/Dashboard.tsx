"use client";

import { useEffect, useState } from "react";
import { useDashboardStore } from "@/store/dashboardStore";
import { format } from "date-fns";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area, ComposedChart, Bar } from "recharts";
import { ShieldAlert, Activity, Cpu, Bell, Gauge, Terminal, Play, Square, Zap, ChevronRight, TriangleAlert } from "lucide-react";

export default function Dashboard() {
  const store = useDashboardStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    let interval: NodeJS.Timeout;
    if (store.isSimulating) {
      interval = setInterval(() => {
        store.tickSimulation();
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [store.isSimulating, store.tickSimulation]);

  if (!mounted) return null;

  const globalHealth = store.models.reduce((acc, m) => acc + m.healthScore, 0) / store.models.length;
  const criticalModels = store.models.filter(m => m.failureRisk === 'critical' || m.failureRisk === 'high');

  const selectedModel = store.selectedModelId ? store.models.find(m => m.id === store.selectedModelId) : null;

  return (
    <div className="h-screen w-full bg-[#0a0a0a] text-zinc-300 font-mono tracking-tight flex flex-col overflow-hidden">
      {/* Header */}
      <header className="min-h-14 lg:h-14 border-b border-zinc-800 bg-[#0f1115] flex flex-col lg:flex-row items-start lg:items-center justify-between px-4 py-2 lg:py-0 shrink-0 gap-3 lg:gap-0 z-20 shadow-md">
        <div className="flex items-center justify-between w-full lg:w-auto gap-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="text-emerald-500 h-5 w-5" />
            <span className="font-bold text-zinc-100 uppercase tracking-widest text-sm truncate">Confidence Sentinel</span>
          </div>
          <Badge variant="outline" className="bg-zinc-900 border-zinc-700 text-[10px] sm:text-xs text-zinc-400 font-mono shrink-0">MVP/LIVE</Badge>
        </div>
        
        <div className="flex items-center justify-between w-full lg:w-auto gap-4 overflow-x-auto pb-1 lg:pb-0 scrollbar-hide">
          <div className="flex items-center gap-2 text-xs shrink-0 bg-zinc-900 px-3 py-1 rounded border border-zinc-800">
            <span className="text-zinc-500">GLOBAL HEALTH:</span>
            <span className={`font-bold ${globalHealth > 85 ? 'text-emerald-400' : globalHealth > 70 ? 'text-amber-400' : 'text-red-500'}`}>
              {globalHealth.toFixed(1)} / 100
            </span>
          </div>
          
          <div className="hidden lg:block h-4 w-px bg-zinc-800" />
          
          <div className="flex items-center gap-2 shrink-0">
            <Button 
              size="sm" 
              variant={store.isSimulating ? "destructive" : "outline"}
              className={`h-8 text-[10px] sm:text-xs shrink-0 transition-all ${store.isSimulating ? 'bg-red-900/80 hover:bg-red-900 text-white' : 'bg-zinc-900 border-zinc-800 hover:bg-zinc-800'}`}
              onClick={store.toggleSimulation}
            >
              {store.isSimulating ? <Square className="w-3 h-3 mr-1" /> : <Play className="w-3 h-3 mr-1" />}
              {store.isSimulating ? 'STOP SIM' : 'SIMULATE DAY'}
            </Button>

            <Button 
              size="sm" 
              className="h-8 text-[10px] sm:text-xs bg-red-950/40 text-red-500 hover:text-red-400 border border-red-900/50 hover:bg-red-900/60 transition-colors shrink-0 font-bold"
              onClick={store.triggerFailureScenario}
            >
              <Zap className="w-3 h-3 sm:mr-1" />
              <span className="hidden sm:inline">TRIGGER FAILURE SCENARIO</span>
              <span className="sm:hidden">FAILURE</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Layout Workspace */}
      <div className="flex flex-col lg:flex-row flex-1 overflow-hidden relative">
        
        {/* Left Sidebar (Models) - Becomes a horizontal scroll row on mobile */}
        <div className="w-full lg:w-64 border-b lg:border-b-0 lg:border-r border-zinc-800 bg-[#0f1115] flex flex-col shrink-0 z-10 transition-all">
          <div className="p-3 border-b border-zinc-800 text-xs font-semibold text-zinc-500 uppercase flex items-center gap-2 shrink-0 bg-zinc-950">
            <Cpu className="w-4 h-4 text-zinc-400" /> ACTIVE ARCHITECTURES
          </div>
          
          {/* List on Desktop, Row on Mobile */}
          <div className="flex-1 overflow-x-auto overflow-y-auto w-full lg:w-auto scrollbar-hide lg:scrollbar-default">
            <div className="flex lg:flex-col gap-2 lg:gap-1 p-3">
              {store.models.map(model => (
                <button
                  key={model.id}
                  onClick={() => store.setSelectedModel(model.id)}
                  className={`w-56 lg:w-full flex-shrink-0 text-left p-3 rounded-lg border transition-all duration-200 ${
                    store.selectedModelId === model.id 
                      ? 'bg-zinc-800 border-zinc-600 shadow-[inset_2px_0_0_0_#10b981] lg:shadow-[inset_4px_0_0_0_#10b981]' 
                      : 'bg-zinc-950 border-zinc-800 hover:bg-zinc-900'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-semibold text-xs sm:text-sm text-zinc-200 truncate pr-2">{model.name}</span>
                    <Badge variant="outline" className={`text-[9px] px-[4px] py-0 h-4 uppercase tracking-wider shrink-0 font-bold ${
                      model.failureRisk === 'critical' ? 'bg-red-950 border-red-800 text-red-500 animate-pulse' :
                      model.failureRisk === 'high' ? 'bg-amber-950 border-amber-800 text-amber-500' :
                      model.failureRisk === 'elevated' ? 'bg-yellow-950/40 border-yellow-800 text-yellow-500' :
                      'bg-emerald-950/30 border-emerald-900 text-emerald-500'
                    }`}>
                      {model.failureRisk}
                    </Badge>
                  </div>
                  <div className="flex justify-between text-[10px] sm:text-xs">
                    <span className="text-zinc-500">Health: <span className="text-zinc-300 font-medium">{model.healthScore.toFixed(0)}</span></span>
                    <span className="text-zinc-500">ECE@75: <span className="text-zinc-300 font-medium">{model.ece75.toFixed(2)}</span></span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Center Area */}
        <div className="flex-1 flex flex-col bg-[#0a0a0a] min-w-0 overflow-auto scrollbar-hide lg:scrollbar-default relative">
          {!selectedModel ? (
            <div className="flex-1 p-4 sm:p-6 flex flex-col gap-4 sm:gap-6 overflow-auto">
              
              {/* Top Global Stats */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 shrink-0">
                <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl overflow-hidden shadow-sm">
                  <CardContent className="p-3 sm:p-4 flex flex-col justify-between h-full">
                    <div className="flex items-center justify-between text-zinc-500 mb-2">
                      <span className="text-[10px] sm:text-xs uppercase font-medium">Avg Health Score</span>
                      <Activity className="w-4 h-4 text-emerald-500 opacity-80" />
                    </div>
                    <span className="text-2xl sm:text-3xl font-black text-zinc-100 font-sans tracking-tight">{globalHealth.toFixed(0)}</span>
                  </CardContent>
                </Card>
                <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl overflow-hidden shadow-sm">
                  <CardContent className="p-3 sm:p-4 flex flex-col justify-between h-full">
                    <div className="flex items-center justify-between text-zinc-500 mb-2">
                      <span className="text-[10px] sm:text-xs uppercase font-medium">Critical Flags</span>
                      <TriangleAlert className={`w-4 h-4 ${criticalModels.length > 0 ? 'text-red-500 animate-pulse' : 'text-zinc-600'}`} />
                    </div>
                    <span className={`text-2xl sm:text-3xl font-black tracking-tight font-sans ${criticalModels.length > 0 ? 'text-red-400' : 'text-zinc-100'}`}>{criticalModels.length}</span>
                  </CardContent>
                </Card>
                <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl overflow-hidden shadow-sm">
                  <CardContent className="p-3 sm:p-4 flex flex-col justify-between h-full">
                    <div className="flex items-center justify-between text-zinc-500 mb-2">
                      <span className="text-[10px] sm:text-xs uppercase font-medium truncate pr-2">Avg Calib Err</span>
                      <Gauge className="w-4 h-4 text-emerald-500 opacity-80 shrink-0" />
                    </div>
                    <span className="text-xl sm:text-3xl font-black text-zinc-100 font-mono tracking-tighter">
                      {(store.models.reduce((acc, m) => acc + m.calibrationError, 0)/store.models.length).toFixed(4)}
                    </span>
                  </CardContent>
                </Card>
                <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl overflow-hidden shadow-sm">
                  <CardContent className="p-3 sm:p-4 flex flex-col justify-between h-full">
                    <div className="flex items-center justify-between text-zinc-500 mb-2">
                      <span className="text-[10px] sm:text-xs uppercase font-medium">System Status</span>
                      <Terminal className="w-4 h-4 text-blue-500 opacity-80" />
                    </div>
                    <span className="text-xs sm:text-sm font-bold text-blue-400 uppercase tracking-widest mt-1">Active</span>
                  </CardContent>
                </Card>
              </div>

              {/* Global Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 shrink-0 lg:h-[300px]">
                <Card className="bg-zinc-900/50 border-zinc-800 rounded-xl flex flex-col overflow-hidden h-64 lg:h-auto">
                  <CardHeader className="p-4 pb-0 items-start">
                    <CardTitle className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Calibration Drift Over Time</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 p-2 pt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={store.chartData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorDrift" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                        <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} minTickGap={20} />
                        <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                        <RechartsTooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '12px' }} itemStyle={{ color: '#10b981' }} />
                        <Area type="monotone" dataKey="calibrationDrift" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorDrift)" isAnimationActive={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
                <Card className="bg-zinc-900/50 border-zinc-800 rounded-xl flex flex-col overflow-hidden h-64 lg:h-auto">
                  <CardHeader className="p-4 pb-0 items-start">
                    <CardTitle className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Gradient Spikes</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 p-2 pt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={store.chartData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                        <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} minTickGap={20} />
                        <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                        <RechartsTooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '12px' }} />
                        <Bar dataKey="gradientSpike" fill="#3b82f6" opacity={0.8} isAnimationActive={false} />
                        <Line type="monotone" dataKey="eceBase" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              {/* Model Comparison Table */}
              <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl flex-1 flex flex-col font-sans overflow-hidden min-h-[300px]">
                <CardHeader className="p-4 shrink-0 bg-zinc-900 border-b border-zinc-800">
                  <CardTitle className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Global Model Portfolio Comparison</CardTitle>
                </CardHeader>
                <CardContent className="p-0 flex-1 overflow-x-auto w-full scrollbar-hide lg:scrollbar-default">
                  <div className="min-w-[800px]">
                    <Table className="text-[13px]">
                      <TableHeader className="bg-zinc-950/50">
                        <TableRow className="border-zinc-800 hover:bg-transparent">
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider w-[180px]">Model</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider">Target App</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider text-right">App Status</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider text-right">Health</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider text-right">Risk</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider text-right">ECE@25</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider text-right">ECE@75</TableHead>
                          <TableHead className="text-zinc-500 h-10 font-bold uppercase tracking-wider text-right">RMSE</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {store.models.map((model) => (
                          <TableRow 
                            key={model.id} 
                            className="border-zinc-800 hover:bg-zinc-800/60 transition-colors cursor-pointer group"
                            onClick={() => store.setSelectedModel(model.id)}
                          >
                            <TableCell className="font-semibold text-zinc-200 group-hover:text-blue-400 transition-colors">{model.name}</TableCell>
                            <TableCell className="text-zinc-400 truncate max-w-[150px]">{model.appContext.name}</TableCell>
                            <TableCell className="text-right whitespace-nowrap">
                              <span className="text-zinc-300 font-mono text-[10px] sm:text-xs">
                                {model.appContext.metricName === 'Latency (ms)' ? model.appContext.metricValue.toFixed(2) : Math.round(model.appContext.metricValue).toLocaleString()} {model.appContext.metricName.split(' ')[0]}
                              </span>
                            </TableCell>
                            <TableCell className="text-right">
                              <span className={`font-mono font-bold ${model.healthScore > 85 ? 'text-emerald-400' : model.healthScore > 70 ? 'text-amber-400' : 'text-red-500'}`}>
                                {model.healthScore.toFixed(0)}
                              </span>
                            </TableCell>
                            <TableCell className="text-right font-bold uppercase text-[11px] tracking-wider">
                              <span className={model.failureRisk === 'critical' ? 'text-red-500' : model.failureRisk === 'high' ? 'text-amber-500' : 'text-zinc-400'}>
                                {model.failureRisk}
                              </span>
                            </TableCell>
                            <TableCell className="text-right text-zinc-400 font-mono">{model.ece25.toFixed(3)}</TableCell>
                            <TableCell className="text-right text-zinc-400 font-mono">{model.ece75.toFixed(3)}</TableCell>
                            <TableCell className="text-right text-zinc-400 font-mono">{model.rmse.toFixed(3)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            // Dedicated Model Detail View
            <div className="flex-1 flex flex-col overflow-auto w-full z-10 bg-[#0a0a0a] min-h-full">
              <div className="p-4 border-b border-zinc-800 flex items-center justify-between shrink-0 bg-[#0f1115] sticky top-0 z-20 shadow-md">
                <div className="flex items-center gap-3">
                  <Button variant="outline" size="sm" className="h-8 w-8 p-0 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 border-zinc-700 bg-zinc-900 rounded-lg transition-all" onClick={() => store.setSelectedModel(null)}>
                    <ChevronRight className="w-5 h-5 rotate-180" />
                  </Button>
                  <h2 className="text-base sm:text-lg font-black text-zinc-100 uppercase tracking-wide truncate">{selectedModel.name} <span className="text-zinc-500 font-medium text-xs sm:text-sm ml-2 font-mono tracking-normal hidden sm:inline">Detail View</span></h2>
                </div>
                <Badge variant="outline" className={`text-[10px] sm:text-xs uppercase font-bold shrink-0 shadow-sm ${
                      selectedModel.failureRisk === 'critical' ? 'bg-red-950/80 border-red-800 text-red-500' :
                      selectedModel.failureRisk === 'high' ? 'bg-amber-950/80 border-amber-800 text-amber-500' :
                      selectedModel.failureRisk === 'elevated' ? 'bg-yellow-950/40 border-yellow-800 text-yellow-500' :
                      'bg-emerald-950/30 border-emerald-900 text-emerald-500'
                    }`}>
                  RISK: {selectedModel.failureRisk}
                </Badge>
              </div>

              <div className="p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 pb-20">
                
                {/* Insights Panel */}
                <div className="col-span-1 lg:col-span-4 flex flex-col gap-4">
                  
                  {selectedModel.failureTimePredicted && (
                    <motion.div 
                      initial={{ scale: 0.95, opacity: 0 }} 
                      animate={{ scale: 1, opacity: 1 }}
                      className="p-5 bg-red-950/30 border-2 border-red-900/50 rounded-xl flex flex-col items-center justify-center text-center shadow-lg relative overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-red-500/5 animate-pulse" />
                      <TriangleAlert className="w-8 h-8 text-red-500 mb-2 relative z-10 opacity-90" />
                      <span className="text-red-400 text-xs font-black uppercase tracking-widest mb-1 relative z-10">Pre-failure signature detected</span>
                      <span className="text-5xl font-black text-red-500 font-sans tracking-tight relative z-10 drop-shadow-md">{selectedModel.failureTimePredicted}</span>
                      <span className="text-zinc-300 text-xs mt-1 relative z-10 font-bold">Timesteps to Degradation</span>
                    </motion.div>
                  )}

                  <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl shadow-sm">
                    <CardHeader className="p-4 pb-2 border-b border-zinc-800/50">
                      <CardTitle className="text-xs text-zinc-400 font-bold uppercase tracking-widest flex items-center gap-2">
                        <Activity className="w-4 h-4 text-zinc-500" /> Target Application & Risk
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4 flex flex-col gap-3">
                      <div className="flex flex-col">
                        <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider mb-1">Use Case</span>
                        <span className="text-zinc-100 text-sm font-medium">{selectedModel.appContext.name}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider mb-1">Live Capital Exposure</span>
                        <span className="text-blue-400 text-xl font-mono">${selectedModel.appContext.exposure.toFixed(1)}M</span>
                      </div>
                      <div className="flex flex-col p-3 bg-zinc-950 rounded border border-zinc-800/50">
                        <div className="flex items-center justify-between">
                          <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider">{selectedModel.appContext.metricName}</span>
                          <span className={`text-xs font-bold uppercase tracking-widest ${selectedModel.failureRisk === 'critical' ? 'text-red-500 animate-pulse' : 'text-emerald-500'}`}>Live</span>
                        </div>
                        <span className={`text-2xl font-mono mt-1 ${selectedModel.failureRisk === 'critical' ? 'text-red-500' : 'text-zinc-100'}`}>
                          {selectedModel.appContext.metricName === 'Latency (ms)' ? selectedModel.appContext.metricValue.toFixed(2) : Math.round(selectedModel.appContext.metricValue).toLocaleString()}
                        </span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl shadow-sm">
                    <CardHeader className="p-4 pb-2 border-b border-zinc-800/50">
                      <CardTitle className="text-xs text-zinc-400 font-bold uppercase tracking-widest flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-zinc-500" /> Auto-Generated Insights
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4">
                      <ul className="space-y-3">
                        {selectedModel.insights.map((insight, idx) => (
                          <li key={idx} className={`flex gap-3 items-start border-l-2 pl-3 py-1 ${insight.includes('failure') || insight.includes('indicates upcoming') ? 'border-red-500' : 'border-blue-500'}`}>
                            {insight.includes('Pre-failure') || insight.includes('indicates upcoming') ? (
                              <span className="text-red-300 text-sm font-medium">{insight}</span>
                            ) : (
                              <span className="text-zinc-300 text-sm font-medium">{insight}</span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl shadow-sm">
                    <CardHeader className="p-4 pb-2 border-b border-zinc-800/50">
                      <CardTitle className="text-xs text-zinc-400 font-bold uppercase tracking-widest">Metrics Summary</CardTitle>
                    </CardHeader>
                    <CardContent className="p-4 grid grid-cols-2 gap-4">
                      <div className="flex flex-col bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                        <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider mb-1">Accuracy</span>
                        <span className="text-zinc-100 text-xl font-mono">{(selectedModel.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex flex-col bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                        <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider mb-1">Calib Err</span>
                        <span className="text-zinc-100 text-xl font-mono">{selectedModel.calibrationError.toFixed(4)}</span>
                      </div>
                      <div className="flex flex-col bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                        <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider mb-1">RMSE</span>
                        <span className="text-zinc-100 text-xl font-mono">{selectedModel.rmse.toFixed(4)}</span>
                      </div>
                      <div className="flex flex-col bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                        <span className="text-zinc-500 text-[10px] font-bold uppercase tracking-wider mb-1">ECE Flat Flag</span>
                        <Badge variant="outline" className={`mt-1 text-[10px] font-bold uppercase w-fit ${selectedModel.eceFlat ? 'text-emerald-400 border-emerald-900 bg-emerald-950/20' : 'text-red-400 border-red-900 bg-red-950/20'}`}>
                          {selectedModel.eceFlat ? 'TRUE' : 'FALSE'}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Charts Area */}
                <div className="col-span-1 lg:col-span-8 flex flex-col gap-6">
                  
                  <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl shadow-sm flex flex-col">
                    <CardHeader className="p-4 sm:p-5 border-b border-zinc-800/50">
                      <CardTitle className="text-sm font-bold text-zinc-100 uppercase tracking-wider flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-500" />
                        Gradient vs Failure Correlation
                      </CardTitle>
                      <CardDescription className="text-xs text-zinc-400 mt-1">Live relationship between gradient norm spikes and Expected Calibration Error.</CardDescription>
                    </CardHeader>
                    <CardContent className="h-64 sm:h-80 p-2 sm:p-4 pt-4">
                       <ResponsiveContainer width="100%" height="100%">
                         <ComposedChart data={store.chartData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                           <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                           <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} minTickGap={20} />
                           <YAxis yAxisId="left" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={[0, 'auto']}/>
                           <YAxis yAxisId="right" orientation="right" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={[0, 'auto']}/>
                           <RechartsTooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '12px' }} />
                           <Bar yAxisId="left" dataKey="gradientSpike" fill={selectedModel.failureRisk === 'critical' ? '#ef4444' : '#3b82f6'} opacity={0.6} isAnimationActive={false} />
                           {/* Using drift as a mock for ECE in the chart */}
                           <Line yAxisId="right" type="monotone" dataKey="calibrationDrift" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                         </ComposedChart>
                       </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl flex flex-row sm:flex-col items-center sm:justify-center p-4 sm:p-6 text-center justify-between shadow-sm">
                      <span className="text-zinc-500 text-[10px] sm:text-xs font-bold uppercase tracking-widest sm:mb-2">ECE@25</span>
                      <span className="text-xl sm:text-3xl text-zinc-100 font-mono font-bold">{selectedModel.ece25.toFixed(4)}</span>
                    </Card>
                    <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl flex flex-row sm:flex-col items-center sm:justify-center p-4 sm:p-6 text-center justify-between shadow-sm">
                      <span className="text-zinc-500 text-[10px] sm:text-xs font-bold uppercase tracking-widest sm:mb-2">ECE@50</span>
                      <span className="text-xl sm:text-3xl text-zinc-100 font-mono font-bold">{selectedModel.ece50.toFixed(4)}</span>
                    </Card>
                    <Card className="bg-zinc-900/80 border-zinc-800 rounded-xl flex flex-row sm:flex-col items-center sm:justify-center p-4 sm:p-6 text-center relative overflow-hidden justify-between shadow-sm">
                      {selectedModel.ece75 > 0.3 && <div className="absolute inset-0 bg-red-500/10 animate-pulse pointer-events-none" />}
                      <span className="text-zinc-500 text-[10px] sm:text-xs font-bold uppercase tracking-widest sm:mb-2 relative z-10">ECE@75</span>
                      <span className={`text-xl sm:text-3xl font-mono font-black relative z-10 ${selectedModel.ece75 > 0.3 ? 'text-red-500 drop-shadow-[0_0_8px_rgba(239,68,68,0.5)]' : 'text-zinc-100'}`}>{selectedModel.ece75.toFixed(4)}</span>
                    </Card>
                  </div>

                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar (Alert Feed) - Slid-in on mobile or full block? Fixed width on Desktop, normal block on mobile bottom */}
        {/* On mobile, we can render it at the bottom of the scroll area or as a hidden drawer. 
            For data density without hiding, let's just make it a flex block that goes to bottom on mobile. */}
        {(!selectedModel || typeof window !== 'undefined' && window.innerWidth > 1024) && (
          <div className="w-full lg:w-72 border-t lg:border-t-0 lg:border-l border-zinc-800 bg-[#0f1115] flex flex-col shrink-0 lg:h-full max-h-[40vh] lg:max-h-none z-10">
            <div className="p-3 border-b border-zinc-800 text-xs font-semibold text-zinc-500 uppercase flex items-center gap-2 shrink-0 bg-zinc-950">
              <Bell className="w-4 h-4 text-zinc-400" /> Live Alert Feed
            </div>
            <ScrollArea className="flex-1 bg-[#0a0a0a]/50">
              <div className="p-3 flex flex-col gap-2">
                <AnimatePresence>
                  {store.alerts.map(alert => (
                    <motion.div
                      key={alert.id}
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      transition={{ duration: 0.2 }}
                      className={`p-3 rounded-lg border text-xs flex flex-col gap-1.5 shadow-sm ${
                        alert.severity === 'critical' ? 'bg-red-950/40 border-red-900/80 shadow-[inset_3px_0_0_0_#ef4444]' :
                        alert.severity === 'warning' ? 'bg-amber-950/30 border-amber-900/60 shadow-[inset_3px_0_0_0_#f59e0b]' :
                        'bg-zinc-900/80 border-zinc-800 shadow-[inset_3px_0_0_0_#3b82f6]'
                      }`}
                    >
                      <div className="flex justify-between items-center text-[10px] text-zinc-500 font-mono">
                        <span className="font-bold">{format(new Date(alert.timestamp), 'HH:mm:ss.SSS')}</span>
                        <Badge variant="outline" className={`text-[8px] px-1 py-0 h-4 border-transparent uppercase tracking-wider ${
                          alert.severity === 'critical' ? 'bg-red-900/50 text-red-300' :
                          alert.severity === 'warning' ? 'bg-amber-900/50 text-amber-300' :
                          'bg-blue-950 text-blue-400'
                        }`}>
                          {alert.severity}
                        </Badge>
                      </div>
                      <span className={`leading-snug font-sans text-[13px] ${alert.severity === 'critical' ? 'text-red-200 font-medium' : 'text-zinc-300'}`}>
                        {alert.message}
                      </span>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </ScrollArea>
          </div>
        )}
      </div>
      
      {/* Banner when critical state exists globally */}
      <AnimatePresence>
        {criticalModels.length > 0 && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-red-950/90 border-t border-red-900 text-red-200 text-[10px] sm:text-xs font-black uppercase p-2 sm:p-3 flex justify-center tracking-widest shrink-0 shadow-[0_-5px_20px_rgba(239,68,68,0.15)] relative z-50 text-center"
          >
            <div className="flex items-center gap-2">
              <TriangleAlert className="w-4 h-4 animate-pulse hidden sm:block" />
              <span>! Confidence reliability degrading — Pre-failure detection active ! Risk prevented logic engaged !</span>
              <TriangleAlert className="w-4 h-4 animate-pulse hidden sm:block" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
