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
  toggleSimulation: () => void;
  triggerFailureScenario: () => void;
  setSelectedModel: (id: string | null) => void;
  tickSimulation: () => void;
}

const INITIAL_MODELS: ModelData[] = [
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

export const useDashboardStore = create<DashboardState>((set, get) => ({
  isSimulating: false,
  currentTimeStep: 0,
  models: INITIAL_MODELS,
  alerts: [
    { id: '1', timestamp: new Date().toISOString(), message: 'System initialization complete. All monitors active.', severity: 'info' }
  ],
  chartData: generateInitialChartData(),
  selectedModelId: null,

  toggleSimulation: () => set((state) => ({ isSimulating: !state.isSimulating })),
  
  triggerFailureScenario: () => {
    // Inject a sudden failure in LSTM
    set((state) => {
      const newModels = state.models.map(m => {
        if (m.id === 'lstm') {
          return {
            ...m,
            healthScore: 35,
            failureRisk: 'critical',
            ece75: 0.45,
            calibrationError: 0.25,
            gradEceCorr: 0.92,
            failureTimePredicted: 37,
            insights: ["Pre-failure signature detected!", "LSTM failures are sudden.", "Gradient spike indicates upcoming failure."],
            appContext: {
              ...m.appContext,
              metricValue: 0 // e.g. trades stopped or massive positional drop
            }
          } as ModelData;
        }
        return m;
      });

      const failureAlert: Alert = {
        id: Math.random().toString(),
        timestamp: new Date().toISOString(),
        message: 'CRITICAL: LSTM sequence model showing pre-failure signature! Predicted failure in 37 timesteps.',
        severity: 'critical'
      };

      return {
        models: newModels,
        alerts: [failureAlert, ...state.alerts].slice(0, 50)
      };
    });
  },

  setSelectedModel: (id) => set({ selectedModelId: id }),

  tickSimulation: () => {
    const { isSimulating, models, chartData, alerts } = get();
    if (!isSimulating) return;

    const newChartData = [...chartData.slice(1)];
    const lastData = chartData[chartData.length - 1];
    
    // Simulate some general drifting
    const nextDrift = Math.max(0, lastData.calibrationDrift + (Math.random() - 0.45) * 0.12);
    const nextGrad = Math.max(0, lastData.gradientSpike + (Math.random() - 0.5) * 0.1);
    
    newChartData.push({
      time: `T-0`, // We can just use an incrementing index if we want, or a real time
      calibrationDrift: nextDrift,
      gradientSpike: nextGrad,
      eceBase: 0.05 + Math.random()*0.02
    });

    // Randomize some metric fluctuations gracefully
    const newModels = models.map(m => {
      // Small jitter
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
