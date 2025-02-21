import socket
import time
import random
import threading
from scapy.all import IP, TCP, send

def generate_normal_traffic():
    """Generates normal-looking traffic patterns"""
    packet = IP(dst="127.0.0.1")/TCP(dport=random.randint(1000, 65535))
    send(packet, verbose=False)
    time.sleep(random.uniform(0.1, 0.5))  # Random delay between packets

def generate_ddos_traffic():
    """Generates high-volume traffic patterns"""
    for _ in range(100):  # Burst of packets
        packet = IP(dst="127.0.0.1")/TCP(dport=random.randint(1000, 65535))
        send(packet, verbose=False)
    time.sleep(0.1)  # Small delay between bursts

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

if __name__ == "__main__":
    traffic_generator() 