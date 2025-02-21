from scapy.all import sniff, wrpcap, TCP, IP
import time
from datetime import datetime
import os

def monitor_traffic():
    """Monitor and save network traffic continuously"""
    # Create output directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(project_root, 'tests', 'network', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create files immediately
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benign_file = os.path.join(output_dir, f"benign_{timestamp}.pcap")
    ddos_file = os.path.join(output_dir, f"ddos_{timestamp}.pcap")
    
    # Create empty PCAP files
    with open(benign_file, 'wb') as f:
        f.write(b'')
    with open(ddos_file, 'wb') as f:
        f.write(b'')
        
    print(f"Created files:")
    print(f"Benign: {benign_file}")
    print(f"DDoS: {ddos_file}")
    
    benign_packets = []
    ddos_packets = []
    
    def packet_handler(packet):
        if IP in packet and TCP in packet:
            # Classify based on source port
            if 1000 <= packet[TCP].sport <= 5000:
                benign_packets.append(packet)
                wrpcap(benign_file, [packet], append=True)
                print(f"Saved benign packet from port {packet[TCP].sport}")
            elif 5001 <= packet[TCP].sport <= 10000:
                ddos_packets.append(packet)
                wrpcap(ddos_file, [packet], append=True)
                print(f"Saved DDoS packet from port {packet[TCP].sport}")
    
    try:
        print("\nStarting capture... Press Ctrl+C to stop")
        sniff(
            iface="lo0",
            filter="tcp and host 127.0.0.1",
            prn=packet_handler
        )
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        # Save final counts
        print(f"\nCapture complete:")
        print(f"Benign packets: {len(benign_packets)}")
        print(f"DDoS packets: {len(ddos_packets)}")
        
        # Verify files exist and have size
        if os.path.exists(benign_file):
            print(f"Benign file size: {os.path.getsize(benign_file)} bytes")
        if os.path.exists(ddos_file):
            print(f"DDoS file size: {os.path.getsize(ddos_file)} bytes")

if __name__ == "__main__":
    monitor_traffic() 