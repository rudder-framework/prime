"""
ORTHON Data Source Connectors

Multiple data source connectors for real-time streaming analysis.
"""

import time
import numpy as np
from typing import Dict, Iterator, Any, List, Optional


class CryptoStreamConnector:
    """Real-time cryptocurrency market data connector."""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        self.base_url = "https://api.binance.com/api/v3"

    def stream(self) -> Iterator[Dict[str, float]]:
        """Stream real-time crypto prices."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests required for crypto connector: pip install requests")

        while True:
            try:
                response = requests.get(f"{self.base_url}/ticker/price", timeout=5)
                data = response.json()

                # Filter to requested symbols
                prices = {}
                for ticker in data:
                    if ticker['symbol'] in self.symbols:
                        prices[ticker['symbol']] = float(ticker['price'])

                if len(prices) >= 2:  # Need multivariate data
                    yield prices

                time.sleep(1)  # 1-second updates

            except Exception as e:
                print(f"Crypto stream error: {e}")
                time.sleep(5)  # Retry after error


class TurbofanSimulator:
    """NASA-inspired turbofan engine simulator with realistic degradation."""

    def __init__(self, degradation_rate: float = 0.001):
        # Initial sensor values (based on NASA C-MAPSS dataset)
        self.sensors = {
            'T2': 518.67,   # Total temperature at fan inlet (R)
            'T24': 612.18,  # Total temperature at LPC outlet (R)
            'T30': 1411.6,  # Total temperature at HPC outlet (R)
            'T50': 1343.8,  # Total temperature at LPT outlet (R)
            'P2': 14.62,    # Pressure at fan inlet (psia)
            'P15': 21.61,   # Total pressure in bypass-duct (psia)
            'P30': 553.9,   # Total pressure at HPC outlet (psia)
            'Nf': 2388.03,  # Physical fan speed (rpm)
            'Nc': 9046.19,  # Physical core speed (rpm)
            'epr': 1.30,    # Engine pressure ratio (P50/P2)
            'Ps30': 1.94,   # Static pressure at HPC outlet (psia)
            'phi': 517.91,  # Ratio of fuel flow to Ps30 (pps/psi)
        }

        self.time_step = 0
        self.degradation_rate = degradation_rate

    def stream(self) -> Iterator[Dict[str, float]]:
        """Stream simulated turbofan sensor data with degradation."""
        while True:
            # Apply degradation and noise
            current_sensors = {}

            for sensor, base_value in self.sensors.items():
                # Gradual degradation
                degradation = self.degradation_rate * self.time_step

                # Sensor-specific degradation patterns
                if 'T' in sensor:  # Temperature sensors
                    degraded_value = base_value + degradation * base_value * 0.002
                    noise_level = base_value * 0.01
                elif 'P' in sensor:  # Pressure sensors
                    degraded_value = base_value + degradation * base_value * 0.001
                    noise_level = base_value * 0.005
                else:  # Other sensors
                    degraded_value = base_value + degradation * base_value * 0.0005
                    noise_level = base_value * 0.002

                # Add realistic noise
                current_sensors[sensor] = (degraded_value +
                                         np.random.normal(0, noise_level))

            yield current_sensors
            self.time_step += 1
            time.sleep(0.1)  # 10Hz simulation

    def reset(self):
        """Reset simulation to initial state."""
        self.time_step = 0


class ChemicalReactorSimulator:
    """Chemical reactor simulator with realistic process dynamics."""

    def __init__(self):
        # Initial state
        self.temperature = 350.0  # K
        self.pressure = 2.5       # bar
        self.concentration = 0.8  # mol/L
        self.flow_rate = 10.0     # L/min
        self.ph = 7.2
        self.conductivity = 80.0  # mS/cm

    def stream(self) -> Iterator[Dict[str, float]]:
        """Stream simulated chemical reactor data."""
        while True:
            dt = 0.1  # Time step

            # Process dynamics (simplified CSTR model)
            # Temperature dynamics
            heat_generation = self.concentration * 1000  # Reaction heat
            heat_removal = 0.1 * (self.temperature - 298)  # Cooling
            dT_dt = (heat_generation - heat_removal) / 5000
            self.temperature += dT_dt * dt + np.random.normal(0, 0.5)

            # Pressure dynamics
            dP_dt = 0.01 * (self.temperature - 350) - 0.05 * (self.pressure - 2.5)
            self.pressure += dP_dt * dt + np.random.normal(0, 0.02)

            # Concentration dynamics (reaction kinetics)
            reaction_rate = 0.001 * self.concentration * np.exp(-1000/self.temperature)
            feed_rate = self.flow_rate * (1.0 - self.concentration) / 100
            dC_dt = feed_rate - reaction_rate
            self.concentration += dC_dt * dt + np.random.normal(0, 0.01)

            # Keep concentration bounded
            self.concentration = np.clip(self.concentration, 0.1, 1.5)

            # Secondary variables
            self.ph = 7.2 + 0.1 * (self.concentration - 0.8) + np.random.normal(0, 0.05)
            self.conductivity = self.concentration * 100 + np.random.normal(0, 1)

            yield {
                'temperature': self.temperature,
                'pressure': self.pressure,
                'concentration': self.concentration,
                'flow_rate': self.flow_rate,
                'ph': self.ph,
                'conductivity': self.conductivity
            }

            time.sleep(dt)

    def reset(self):
        """Reset simulation to initial state."""
        self.temperature = 350.0
        self.pressure = 2.5
        self.concentration = 0.8
        self.flow_rate = 10.0
        self.ph = 7.2
        self.conductivity = 80.0


class SystemMetricsConnector:
    """Local system performance metrics connector."""

    def stream(self) -> Iterator[Dict[str, float]]:
        """Stream system performance metrics."""
        try:
            import psutil
        except ImportError:
            raise ImportError("psutil required for system metrics: pip install psutil")

        # Initialize counters for delta calculations
        last_disk_read = psutil.disk_io_counters().read_bytes
        last_disk_write = psutil.disk_io_counters().write_bytes
        last_net_sent = psutil.net_io_counters().bytes_sent
        last_net_recv = psutil.net_io_counters().bytes_recv

        while True:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent

                # Disk I/O (rate)
                disk_counters = psutil.disk_io_counters()
                disk_read_rate = (disk_counters.read_bytes - last_disk_read) / 1024 / 1024  # MB/s
                disk_write_rate = (disk_counters.write_bytes - last_disk_write) / 1024 / 1024
                last_disk_read = disk_counters.read_bytes
                last_disk_write = disk_counters.write_bytes

                # Network I/O (rate)
                net_counters = psutil.net_io_counters()
                net_sent_rate = (net_counters.bytes_sent - last_net_sent) / 1024 / 1024  # MB/s
                net_recv_rate = (net_counters.bytes_recv - last_net_recv) / 1024 / 1024
                last_net_sent = net_counters.bytes_sent
                last_net_recv = net_counters.bytes_recv

                yield {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_read_rate': max(0, disk_read_rate),
                    'disk_write_rate': max(0, disk_write_rate),
                    'network_sent_rate': max(0, net_sent_rate),
                    'network_recv_rate': max(0, net_recv_rate),
                }
            except Exception as e:
                print(f"System metrics error: {e}")

            time.sleep(1)


class SyntheticDataSource:
    """Synthetic data source for testing with configurable patterns."""

    def __init__(self,
                 num_signals: int = 5,
                 pattern: str = 'normal',
                 noise_level: float = 0.1):
        self.num_signals = num_signals
        self.pattern = pattern
        self.noise_level = noise_level
        self.time_step = 0

    def stream(self) -> Iterator[Dict[str, float]]:
        """Stream synthetic data with various patterns."""
        while True:
            t = self.time_step * 0.1  # Time in seconds

            if self.pattern == 'normal':
                # Normal operation - stable multivariate
                values = {
                    f'signal_{i}': np.sin(0.1 * t + i) + self.noise_level * np.random.randn()
                    for i in range(self.num_signals)
                }
            elif self.pattern == 'degradation':
                # Gradual degradation - increasing coupling
                coupling = min(0.9, 0.1 + 0.001 * self.time_step)
                base = np.sin(0.1 * t)
                values = {
                    f'signal_{i}': (1-coupling) * np.sin(0.1*t + i) + coupling * base + self.noise_level * np.random.randn()
                    for i in range(self.num_signals)
                }
            elif self.pattern == 'collapse':
                # Dimensional collapse after time threshold
                if self.time_step < 500:
                    # Normal operation
                    values = {
                        f'signal_{i}': np.sin(0.1 * t + i) + self.noise_level * np.random.randn()
                        for i in range(self.num_signals)
                    }
                else:
                    # Collapse - all signals correlated
                    base = np.sin(0.1 * t)
                    values = {
                        f'signal_{i}': base + 0.01 * i + self.noise_level * np.random.randn()
                        for i in range(self.num_signals)
                    }
            elif self.pattern == 'oscillating':
                # Oscillating behavior
                values = {
                    f'signal_{i}': np.sin(0.5 * t + i) * np.sin(0.05 * t) + self.noise_level * np.random.randn()
                    for i in range(self.num_signals)
                }
            else:
                # Random walk
                if self.time_step == 0:
                    self._random_state = np.zeros(self.num_signals)
                self._random_state += 0.1 * np.random.randn(self.num_signals)
                values = {
                    f'signal_{i}': self._random_state[i]
                    for i in range(self.num_signals)
                }

            yield values
            self.time_step += 1
            time.sleep(0.1)

    def reset(self):
        """Reset simulation."""
        self.time_step = 0
        if hasattr(self, '_random_state'):
            del self._random_state


# Data source registry
DATA_SOURCES = {
    'crypto': {
        'class': CryptoStreamConnector,
        'description': 'Real-time cryptocurrency market data (Binance API)',
        'update_interval': 1.0,
    },
    'turbofan': {
        'class': TurbofanSimulator,
        'description': 'NASA C-MAPSS inspired turbofan engine simulation',
        'update_interval': 0.1,
    },
    'reactor': {
        'class': ChemicalReactorSimulator,
        'description': 'Chemical reactor (CSTR) process simulation',
        'update_interval': 0.1,
    },
    'system': {
        'class': SystemMetricsConnector,
        'description': 'Local system performance metrics (CPU, memory, disk, network)',
        'update_interval': 1.0,
    },
    'synthetic': {
        'class': SyntheticDataSource,
        'description': 'Synthetic data for testing (normal, degradation, collapse patterns)',
        'update_interval': 0.1,
    },
}


def get_stream_connector(source_type: str, **kwargs):
    """Get a data stream connector by type."""
    if source_type not in DATA_SOURCES:
        available = ", ".join(DATA_SOURCES.keys())
        raise ValueError(f"Unknown source type: {source_type}. Available: {available}")

    source_info = DATA_SOURCES[source_type]
    connector = source_info['class'](**kwargs)

    # Attach update_interval to connector
    connector.update_interval = source_info.get('update_interval', 0.1)

    return connector


def get_source_info(source_type: str) -> Dict[str, Any]:
    """Get info about a data source."""
    if source_type not in DATA_SOURCES:
        available = ", ".join(DATA_SOURCES.keys())
        raise ValueError(f"Unknown source type: {source_type}. Available: {available}")

    return DATA_SOURCES[source_type]
