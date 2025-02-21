import socket
import time
import random
import threading
from scapy.all import Ether, IP, TCP, sendp

def generate_normal_traffic():
    """Generates normal-looking traffic patterns"""
    # Use source port 1000-5000 for benign traffic
    packet = Ether()/IP(src="127.0.0.1", dst="127.0.0.1")/TCP(
        sport=random.randint(1000, 5000),
        dport=random.randint(1000, 65535)
    )
    packet[TCP].flags = "S"  # SYN flag
    try:
        sendp(packet, iface="lo0", verbose=False)
        print("Sent BENIGN packet")
    except Exception as e:
        print(f"Error sending BENIGN packet: {e}")
    time.sleep(random.uniform(0.1, 0.5))

def generate_ddos_traffic():
    """Generates high-volume traffic patterns"""
    for _ in range(100):
        # Use source port 5001-10000 for DDoS traffic
        packet = Ether()/IP(src="127.0.0.1", dst="127.0.0.1")/TCP(
            sport=random.randint(5001, 10000),
            dport=random.randint(1000, 65535)
        )
        packet[TCP].flags = "S"  # SYN flag
        try:
            sendp(packet, iface="lo0", verbose=False)
            print("Sent DDoS packet")
        except Exception as e:
            print(f"Error sending DDoS packet: {e}")
    time.sleep(0.1)

def traffic_generator():
    print("🚀 Starting Traffic Generator")
    while True:
        try:
            # Alternate between normal and DDoS patterns
            print("📊 Generating normal traffic...")
            for _ in range(10):
                generate_normal_traffic()
            
            print("🔥 Generating DDoS-like traffic...")
            for _ in range(5):
                generate_ddos_traffic()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping Traffic Generator")
            break

def packet_handler(packet):
    print(packet.summary())

if __name__ == "__main__":
    traffic_generator() 