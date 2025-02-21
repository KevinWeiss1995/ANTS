from scapy.all import sniff, wrpcap
import time
from datetime import datetime
import os

def packet_callback(packet):
    """Callback function to process each captured packet"""
    print(f"Captured packet: {packet.summary()}")

def monitor_traffic(duration=60):
    """
    Monitor and save network traffic
    
    Args:
        duration: How long to capture (in seconds)
    """
    # Set up the correct data directory path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(project_root, 'tests', 'network', 'data')
    
    # Determine correct interface name based on OS
    if os.name == 'posix':  # macOS or Linux
        import platform
        if platform.system() == 'Darwin':  # macOS
            interface = 'lo0'
        else:  # Linux
            interface = 'lo'
    else:  # Windows
        interface = 'loopback'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"capture_{timestamp}.pcap")
    
    print(f"📡 Starting traffic capture for {duration} seconds...")
    print(f"💾 Saving to: {filename}")
    print(f"🔌 Using interface: {interface}")
    
    # Capture packets
    packets = sniff(
        iface=interface,
        prn=packet_callback,
        timeout=duration
    )
    
    # Save captured packets
    wrpcap(filename, packets)
    print(f"✅ Captured {len(packets)} packets")

if __name__ == "__main__":
    try:
        while True:
            monitor_traffic()
    except KeyboardInterrupt:
        print("\n🛑 Stopping traffic monitor") 