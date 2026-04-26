"use client";

import { create } from 'zustand';

export type AlertSeverity = 'info' | 'warning' | 'critical';

export interface Alert {
  id: string;
  timestamp: string;
  message: string;
  severity: AlertSeverity;
}

export interface ModelData {
  id: string;
  name: string;
  healthScore: number; // 0-100
  failureRisk: 'low' | 'elevated' | 'high' | 'critical';
  ece25: number;
  ece50: number;
  ece75: number;
  calibrationError: number;
  rmse: number;
  rSquare: number;
  accuracy: number;
  gradEceCorr: number;
  eceFlat: boolean;
  failureTimePredicted: number | null; // e.g. 37 timesteps
  insights: string[];
  appContext: {
    name: string;
    exposure: number; // In millions
    metricName: string;
    metricValue: number; // Dynamic value
  };
}

export interface ChartDataPoint {
  time: string;
  calibrationDrift: number;
  gradientSpike: number;
  eceBase: number;
  [key: string]: string | number; // For dynamic recharts parsing
}

interface DashboardState {
  isSimulating: boolean;
  currentTimeStep: number;
  models: ModelData[];
  alerts: Alert[];
  chartData: ChartDataPoint[];
  selectedModelId: string | null;
  isLoading: boolean;
  error: string | null;
  ws: WebSocket | null;
  
  // API Methods
  fetchDashboardState: () => Promise<void>;
  fetchModels: () => Promise<void>;
  fetchModelDetail: (modelId: string) => Promise<ModelData | null>;
  
  // WebSocket
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;
  
  // UI Actions
  toggleSimulation: () => void;
  triggerFailureScenario: () => Promise<void>;
  setSelectedModel: (id: string | null) => void;
  tickSimulation: () => void;
  
  // Real-time updates from backend
  updateMetrics: (data: any) => void;
  addAlert: (alert: Alert) => void;
}

// Fallback data for when backend is unavailable
const FALLBACK_MODELS: ModelData[] = [
  {
    id: 'deep-mlp', name: 'Deep MLP Core', healthScore: 92, failureRisk: 'low',
    ece25: 0.02, ece50: 0.05, ece75: 0.08, calibrationError: 0.04, rmse: 0.12, rSquare: 0.89, accuracy: 0.91, gradEceCorr: 0.1, eceFlat: true, failureTimePredicted: null,
    insights: ["Model shows stable calibration."],
    appContext: { name: "HFT Options Arbitrage", exposure: 14.2, metricName: "Trades/sec", metricValue: 1402 }
  },
  {
    id: 'lstm', name: 'LSTM Sequence', healthScore: 85, failureRisk: 'elevated',
    ece25: 0.04, ece50: 0.09, ece75: 0.15, calibrationError: 0.07, rmse: 0.18, rSquare: 0.81, accuracy: 0.85, gradEceCorr: 0.4, eceFlat: false, failureTimePredicted: null,
    insights: ["LSTM variance increasing."],
    appContext: { name: "NatGas Futures Prediction", exposure: 42.5, metricName: "Active Positions", metricValue: 840 }
  },
  {
    id: 'mlp', name: 'Standard MLP', healthScore: 95, failureRisk: 'low',
    ece25: 0.01, ece50: 0.03, ece75: 0.06, calibrationError: 0.03, rmse: 0.11, rSquare: 0.92, accuracy: 0.94, gradEceCorr: 0.05, eceFlat: true, failureTimePredicted: null,
    insights: ["Stable performance."],
    appContext: { name: "Equity Block Trading", exposure: 110.1, metricName: "Volume (Shares)", metricValue: 154000 }
  },
  {
    id: 'res-mlp', name: 'Residual MLP', healthScore: 88, failureRisk: 'low',
    ece25: 0.03, ece50: 0.06, ece75: 0.09, calibrationError: 0.05, rmse: 0.14, rSquare: 0.85, accuracy: 0.88, gradEceCorr: 0.2, eceFlat: true, failureTimePredicted: null,
    insights: ["Normal operational bounds."],
    appContext: { name: "FX Cross-currency Engine", exposure: 53.4, metricName: "Latency (ms)", metricValue: 0.8 }
  },
  {
    id: 'vanilla-rnn', name: 'Vanilla RNN', healthScore: 78, failureRisk: 'high',
    ece25: 0.09, ece50: 0.15, ece75: 0.25, calibrationError: 0.12, rmse: 0.25, rSquare: 0.73, accuracy: 0.77, gradEceCorr: 0.65, eceFlat: false, failureTimePredicted: 85,
    insights: ["RNN shows gradual degradation.", "Gradient spike indicates upcoming failure."],
    appContext: { name: "Credit Risk Pricing", exposure: 12.8, metricName: "Quotes/sec", metricValue: 345 }
  }
];

const generateInitialChartData = (): ChartDataPoint[] => {
  const data: ChartDataPoint[] = [];
  let baseDrift = 1.0;
  let baseGrad = 0.5;
  for(let i=0; i<30; i++) {
    baseDrift += (Math.random() - 0.4) * 0.1;
    baseGrad += (Math.random() - 0.5) * 0.08;
    data.push({
      time: `T-${30 - i}`,
      calibrationDrift: Math.max(0, baseDrift),
      gradientSpike: Math.max(0, Math.sin(i*0.2) * 0.1 + baseGrad),
      eceBase: 0.05 + Math.random()*0.02
    });
  }
  return data;
};

const API_URL = typeof window !== 'undefined' 
  ? process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

const WS_URL = typeof window !== 'undefined'
  ? process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  : 'ws://localhost:8000';

export const useDashboardStore = create<DashboardState>((set, get) => ({
  isSimulating: false,
  currentTimeStep: 0,
  models: FALLBACK_MODELS,
  alerts: [
    { id: '1', timestamp: new Date().toISOString(), message: 'System initialization complete. All monitors active.', severity: 'info' }
  ],
  chartData: generateInitialChartData(),
  selectedModelId: null,
  isLoading: false,
  error: null,
  ws: null,

  // API Methods
  fetchDashboardState: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch(`${API_URL}/api/dashboard`);
      if (!response.ok) throw new Error('Failed to fetch dashboard state');
      
      const data = await response.json();
      const models = data.models.map((m: any) => ({
        id: m.id,
        name: m.name,
        healthScore: m.health_score,
        failureRisk: m.failure_risk,
        ece25: m.ece_25,
        ece50: m.ece_50,
        ece75: m.ece_75,
        calibrationError: m.calibration_error,
        rmse: m.rmse,
        rSquare: m.r_square,
        accuracy: m.accuracy,
        gradEceCorr: m.grad_ece_correlation,
        eceFlat: m.ece_flat,
        failureTimePredicted: m.failure_time_predicted,
        insights: m.insights || [],
        appContext: {
          name: m.app_context.name,
          exposure: m.app_context.exposure,
          metricName: m.app_context.metric_name,
          metricValue: m.app_context.metric_value,
        }
      }));

      const alerts = data.alerts.map((a: any) => ({
        id: a.id,
        timestamp: a.timestamp,
        message: a.message,
        severity: a.severity as AlertSeverity,
      }));

      set({ models, alerts, isLoading: false });
      get().connectWebSocket();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      set({ error: message, isLoading: false });
      console.error('Failed to fetch dashboard:', message);
    }
  },

  fetchModels: async () => {
    try {
      const response = await fetch(`${API_URL}/api/models`);
      if (!response.ok) throw new Error('Failed to fetch models');
      
      const data = await response.json();
      const models = data.map((m: any) => ({
        ...get().models.find(mod => mod.id === m.id),
        healthScore: m.health_score,
        failureRisk: m.failure_risk,
      }));

      set({ models });
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  },

  fetchModelDetail: async (modelId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/models/${modelId}`);
      if (!response.ok) throw new Error('Failed to fetch model detail');
      
      const data = await response.json();
      return {
        id: data.id,
        name: data.name,
        healthScore: data.health_score,
        failureRisk: data.failure_risk,
        ece25: data.ece_25,
        ece50: data.ece_50,
        ece75: data.ece_75,
        calibrationError: data.calibration_error,
        rmse: data.rmse,
        rSquare: data.r_square,
        accuracy: data.accuracy,
        gradEceCorr: data.grad_ece_correlation,
        eceFlat: data.ece_flat,
        failureTimePredicted: data.failure_time_predicted,
        insights: data.insights || [],
        appContext: {
          name: data.app_context.name,
          exposure: data.app_context.exposure,
          metricName: data.app_context.metric_name,
          metricValue: data.app_context.metric_value,
        }
      } as ModelData;
    } catch (error) {
      console.error('Failed to fetch model detail:', error);
      return null;
    }
  },

  connectWebSocket: () => {
    if (typeof window === 'undefined') return;
    
    try {
      const ws = new WebSocket(`${WS_URL}/ws/metrics`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        set({ ws });
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'metrics_update') {
          get().updateMetrics(message.data);
        } else if (message.type === 'alert') {
          get().addAlert({
            id: message.data.id,
            timestamp: message.data.timestamp,
            message: message.data.message,
            severity: message.data.severity as AlertSeverity,
          });
        } else if (message.type === 'failure_scenario') {
          get().updateMetrics(message.data);
          get().addAlert({
            id: message.data.alert_id,
            timestamp: new Date().toISOString(),
            message: `CRITICAL: ${message.data.model_id} model showing pre-failure signature!`,
            severity: 'critical',
          });
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        set({ error: 'WebSocket connection failed' });
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        set({ ws: null });
        // Attempt to reconnect after 3 seconds
        setTimeout(() => get().connectWebSocket(), 3000);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  },

  disconnectWebSocket: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null });
    }
  },

  toggleSimulation: () => set((state) => ({ isSimulating: !state.isSimulating })),

  triggerFailureScenario: async () => {
    try {
      const response = await fetch(`${API_URL}/api/simulate/trigger-failure?model_id=lstm`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to trigger failure scenario');
      console.log('Failure scenario triggered');
    } catch (error) {
      console.error('Failed to trigger failure scenario:', error);
      set({ error: 'Failed to trigger failure scenario' });
    }
  },

  setSelectedModel: (id) => set({ selectedModelId: id }),

  updateMetrics: (data: any) => {
    const models = get().models.map(m => {
      if (m.id === data.model_id) {
        return {
          ...m,
          healthScore: data.health_score ?? m.healthScore,
          failureRisk: data.failure_risk ?? m.failureRisk,
        };
      }
      return m;
    });
    set({ models });
  },

  addAlert: (alert: Alert) => {
    set((state) => ({
      alerts: [alert, ...state.alerts].slice(0, 50),
    }));
  },

  tickSimulation: () => {
    const { isSimulating, models, chartData, alerts } = get();
    if (!isSimulating) return;

    const newChartData = [...chartData.slice(1)];
    const lastData = chartData[chartData.length - 1];
    
    // Simulate some general drifting
    const nextDrift = Math.max(0, lastData.calibrationDrift + (Math.random() - 0.45) * 0.12);
    const nextGrad = Math.max(0, lastData.gradientSpike + (Math.random() - 0.5) * 0.1);
    
    newChartData.push({
      time: `T-0`,
      calibrationDrift: nextDrift,
      gradientSpike: nextGrad,
      eceBase: 0.05 + Math.random()*0.02
    });

    // Randomize some metric fluctuations gracefully
    const newModels = models.map(m => {
      const jitter = (Math.random() - 0.5) * 2;
      return {
        ...m,
        healthScore: Math.min(100, Math.max(0, m.healthScore + jitter * 0.5)),
        ece50: Math.max(0, m.ece50 + (Math.random() - 0.5) * 0.005),
        appContext: {
          ...m.appContext,
          metricValue: m.appContext.metricName === 'Latency (ms)' ? 
                         Math.max(0.1, m.appContext.metricValue + (Math.random() - 0.5) * 0.1) :
                         Math.max(0, m.appContext.metricValue + Math.floor((Math.random() - 0.5) * 15))
        }
      };
    });

    set({
      currentTimeStep: get().currentTimeStep + 1,
      chartData: newChartData,
      models: newModels
    });
  }
}));
