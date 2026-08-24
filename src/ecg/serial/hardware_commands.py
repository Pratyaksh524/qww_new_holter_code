"""
ECG Hardware Command Protocol Handler

Protocol Specification:
- Start Byte: 0xE8
- Counter: 0x00 (byte 2)
- Length: 0x11 (17 bytes, byte 3)
- OpCode/Code: Operation code (byte 4)
- Checksum: 0x00 (byte 5)
- Data: Bytes 6-21 (16 bytes)
- End Byte: 0x8E (byte 22)

Command OpCodes:
- 0x10: Start command
- 0x11: Stop command
- 0x13: Machine serial number check
- 0x14: Version check
- 0x15: Software close

Response Codes:
- 0x21: ACK (acknowledgment)
- 0x23: Machine serial number data response
- 0x24: Actual data (for version response)
"""

import time
from typing import Optional, Dict, Tuple

# Protocol constants
START_BYTE = 0xE8
END_BYTE = 0x8E
PACKET_LENGTH = 0x11  # 17 bytes
FRAME_LEN = 22  # Total frame length (22 bytes)
ACK_CODE = 0x21
DATA_CODE_VERSION = 0x24  # Version data response
DATA_CODE_MACHINE_SERIAL = 0x23  # Machine serial number data response

# Command OpCodes
OPCODE_START = 0x10
OPCODE_STOP = 0x11
OPCODE_MACHINE_SERIAL = 0x13
OPCODE_VERSION = 0x14
OPCODE_CLOSE = 0x15

# Timeout for waiting for responses (seconds)
RESPONSE_TIMEOUT = 0.5  # Reduced from 2.0 for instant connection


class HardwareCommandHandler:
    """Handle hardware command protocol for ECG device"""
    
    def __init__(self, serial_port):
        """
        Initialize command handler
        
        Args:
            serial_port: pyserial Serial object
        """
        self.ser = serial_port
        self.counter = 0x00
    
    def _format_packet_details(self, packet: bytes, direction: str, command_name: str) -> str:
        """
        Format packet details for logging
        
        Args:
            packet: Packet bytes (22 bytes)
            direction: "SEND" or "RECV"
            command_name: Name of the command
            
        Returns:
            str: Formatted packet details
        """
        if len(packet) != 22:
            return f"Invalid packet length: {len(packet)} bytes"
        
        details = []
        details.append(f"\n{'='*70}")
        details.append(f"{direction}: {command_name} Packet")
        details.append(f"{'='*70}")
        details.append(f"Raw Hex: {packet.hex().upper()}")
        details.append(f"Byte-by-byte breakdown:")
        details.append(f"  [0] Start Byte:     0x{packet[0]:02X} ({'Y' if packet[0] == START_BYTE else 'N'})")
        details.append(f"  [1] Counter:       0x{packet[1]:02X} ({packet[1]})")
        details.append(f"  [2] Length:        0x{packet[2]:02X} ({packet[2]} bytes)")
        details.append(f"  [3] OpCode/Code:    0x{packet[3]:02X} ({self._get_code_name(packet[3])})")
        details.append(f"  [4] Checksum:       0x{packet[4]:02X}")
        details.append(f"  [5-20] Data:        {packet[5:21].hex().upper()}")
        if packet[3] == ACK_CODE and len(packet) > 5:
            details.append(f"      └─ Echoed OpCode: 0x{packet[5]:02X} ({self._get_code_name(packet[5])})")
        details.append(f"  [21] End Byte:      0x{packet[21]:02X} ({'Y' if packet[21] == END_BYTE else 'N'})")
        details.append(f"{'='*70}\n")
        return "\n".join(details)
    
    def _get_code_name(self, code: int) -> str:
        """Get human-readable name for OpCode/Response code"""
        code_map = {
            0x10: "START",
            0x11: "STOP",
            0x13: "MACHINE_SERIAL",
            0x14: "VERSION",
            0x15: "CLOSE",
            0x21: "ACK",
            0x23: "MACHINE_SERIAL_DATA",
            0x24: "VERSION_DATA"
        }
        return code_map.get(code, f"UNKNOWN(0x{code:02X})")
    
    def _build_command_packet(self, opcode: int) -> bytes:
        """
        Build a command packet according to protocol
        
        Args:
            opcode: Operation code (0x10, 0x11, 0x13, 0x15)
            
        Returns:
            bytes: Complete command packet (22 bytes)
        """
        packet = bytearray(22)
        packet[0] = START_BYTE      # Byte 0: Start byte
        packet[1] = self.counter    # Byte 1: Counter
        packet[2] = PACKET_LENGTH   # Byte 2: Length (0x11 = 17)
        packet[3] = opcode          # Byte 3: OpCode
        packet[4] = 0x00            # Byte 4: Checksum (currently 0x00)
        # Bytes 5-20: Data (all zeros for commands)
        for i in range(5, 21):
            packet[i] = 0x00
        packet[21] = END_BYTE       # Byte 21: End byte
        
        # Increment counter (wrap at 0x3F = 63)
        self.counter = (self.counter + 1) & 0x3F
        
        return bytes(packet)
    
    def _read_response(self, timeout: float = RESPONSE_TIMEOUT) -> Optional[bytes]:
        """
        Read response packet from device
        
        Args:
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            bytes: Response packet or None if timeout/error
        """
        start_time = time.time()
        buffer = bytearray()
        
        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                buffer.extend(chunk)
                
                # Look for complete packet (START_BYTE ... END_BYTE, 22 bytes)
                start_idx = buffer.find(START_BYTE)
                if start_idx >= 0:
                    # Check if we have enough bytes for a complete packet
                    if len(buffer) >= start_idx + 22:
                        packet = bytes(buffer[start_idx:start_idx + 22])
                        if packet[-1] == END_BYTE and len(packet) == 22:
                            # Remove processed packet from buffer
                            buffer = buffer[start_idx + 22:]
                            return packet
                # Bytes were consumed this pass: look again before sleeping.
                continue

            # Nothing at the port — yield briefly rather than spinning.
            time.sleep(0.002)
        
        return None
    
    def _parse_response(self, packet: bytes) -> Dict[str, any]:
        """
        Parse response packet
        
        Args:
            packet: Response packet bytes (22 bytes)
            
        Returns:
            dict: Parsed response with keys: type, counter, length, code, opcode, data
        """
        if len(packet) != 22 or packet[0] != START_BYTE or packet[21] != END_BYTE:
            return {"type": "invalid", "error": "Invalid packet format"}
        
        response = {
            "type": "unknown",
            "counter": packet[1],
            "length": packet[2],
            "code": packet[3],
            "checksum": packet[4],
            "data": packet[5:21],  # Bytes 5-20 (16 bytes)
        }
        
        # Handle standard ACK code (0x21)
        if packet[3] == ACK_CODE:
            response["type"] = "ack"
            response["opcode"] = packet[5]  # Echoed OpCode in byte 5
        # Handle Version DATA code (0x24)
        elif packet[3] == DATA_CODE_VERSION:
            response["type"] = "version_data"
        # Handle Machine Serial DATA code (0x23)
        elif packet[3] == DATA_CODE_MACHINE_SERIAL:
            response["type"] = "machine_serial_data"
        # Handle device-specific code (0x20) - device uses this for both ACK and DATA
        elif packet[3] == 0x20:
            # Check byte 5 to determine if it's ACK (contains echoed OpCode) or DATA
            # For ACK: byte 5 should contain the echoed OpCode (0x10, 0x11, 0x14, 0x15)
            # For DATA: byte 5 might be 0x00 or start of data
            if packet[5] in [OPCODE_START, OPCODE_STOP, OPCODE_VERSION, OPCODE_CLOSE]:
                response["type"] = "ack"
                response["opcode"] = packet[5]  # Echoed OpCode in byte 5
            else:
                # Likely a DATA packet
                response["type"] = "data"
        else:
            response["type"] = "unknown"
            response["opcode"] = packet[3]
        
        return response
    
    def send_start_command(self, timeout: Optional[float] = None, quiet: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        Send Start command (OpCode 0x10)
        
        Args:
            timeout: Optional timeout in seconds (default: RESPONSE_TIMEOUT)
            quiet: If True, reduce verbosity (useful for port scanning)
        
        Returns:
            tuple: (success: bool, response: dict or None)
        """
        if timeout is None:
            timeout = RESPONSE_TIMEOUT
        
        if not quiet:
            print("\n" + "="*70)
            print("[START] START COMMAND: Initiating communication with device")
            print("="*70)
        
        try:
            # Build and send command
            cmd_packet = self._build_command_packet(OPCODE_START)
            
            # Log what SOFTWARE is SENDING
            if not quiet:
                print(self._format_packet_details(cmd_packet, "SOFTWARE -> DEVICE", "START Command"))
                print(f"[SEND] Software sending START command to device...")
            
            self.ser.write(cmd_packet)
            self.ser.flush()
            
            if not quiet:
                print(f"[OK] Command packet written to serial port ({len(cmd_packet)} bytes)")
                print(f"[WAIT] Waiting for device response (timeout: {timeout}s)...")
            
            # Wait for ACK
            response_packet = self._read_response(timeout=timeout)
            if response_packet:
                # Log what DEVICE is SENDING BACK
                if not quiet:
                    print(self._format_packet_details(response_packet, "DEVICE -> SOFTWARE", "START ACK Response"))
                    print(f"[RECV] Device responded with packet ({len(response_packet)} bytes)")
                
                response = self._parse_response(response_packet)
                if not quiet:
                    print(f"[PARSED] Parsed response: {response}")
                elif quiet:
                    # In quiet mode, still show minimal info for debugging
                    print(f"   [RECV] Response: Code=0x{response.get('code', 0):02X}, Type={response.get('type')}, OpCode=0x{response.get('opcode', 0):02X}")
                
                is_ack = False
                if response["type"] == "ack" and response.get("opcode") == OPCODE_START:
                    is_ack = True
                elif response.get("code") == 0x20:
                    is_ack = True
                    
                if is_ack:
                    if not quiet:
                        print(f"[OK] START COMMAND: Success! Device acknowledged START command")
                        print(f"   ACK OpCode: 0x{response.get('opcode', 0):02X} ({self._get_code_name(response.get('opcode', 0))})")
                        print("="*70 + "\n")
                    return True, response
                else:
                    if not quiet:
                        print(f"[WARN] START COMMAND: Unexpected response type or OpCode mismatch")
                        print(f"   Expected: ACK with OpCode 0x{OPCODE_START:02X}")
                        print(f"   Received: {response}")
                        print("="*70 + "\n")
                    elif quiet:
                        # In quiet mode, show why it failed
                        print(f"   [ERROR] Failed: Expected ACK with OpCode 0x{OPCODE_START:02X}, got {response.get('type')} with OpCode 0x{response.get('opcode', 0):02X}")
                    return False, response
            else:
                if not quiet:
                    print("[ERROR] START COMMAND: No response received from device (timeout)")
                    print(f"   Device did not respond within {timeout} seconds")
                    print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            if not quiet:
                print(f"[ERROR] START COMMAND: Error occurred: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                print("="*70 + "\n")
            return False, None
    
    def send_stop_command(self) -> Tuple[bool, Optional[Dict]]:
        """
        Send Stop command (OpCode 0x11)
        
        Returns:
            tuple: (success: bool, response: dict or None)
        """
        print("\n" + "="*70)
        print("[STOP] STOP COMMAND: Requesting device to stop")
        print("="*70)
        
        try:
            # Build and send command
            cmd_packet = self._build_command_packet(OPCODE_STOP)
            
            # Log what SOFTWARE is SENDING
            print(self._format_packet_details(cmd_packet, "SOFTWARE -> DEVICE", "STOP Command"))
            print(f"[SEND] Software sending STOP command to device...")
            
            self.ser.write(cmd_packet)
            self.ser.flush()
            
            print(f"[OK] Command packet written to serial port ({len(cmd_packet)} bytes)")
            print(f"[WAIT] Waiting for device response (timeout: {RESPONSE_TIMEOUT}s)...")
            
            # Wait for ACK
            response_packet = self._read_response()
            if response_packet:
                # Log what DEVICE is SENDING BACK
                print(self._format_packet_details(response_packet, "DEVICE -> SOFTWARE", "STOP ACK Response"))
                print(f"[RECV] Device responded with packet ({len(response_packet)} bytes)")
                
                response = self._parse_response(response_packet)
                print(f"[PARSED] Parsed response: {response}")
                
                is_ack = False
                if response["type"] == "ack" and response.get("opcode") == OPCODE_STOP:
                    is_ack = True
                elif response.get("code") == 0x20:
                    is_ack = True
                    
                if is_ack:
                    print(f"[OK] STOP COMMAND: Success! Device acknowledged STOP command")
                    print(f"   ACK OpCode: 0x{response.get('opcode', 0):02X} ({self._get_code_name(response.get('opcode', 0))})")
                    print("="*70 + "\n")
                    return True, response
                else:
                    print(f"[WARN] STOP COMMAND: Unexpected response type or OpCode mismatch")
                    print(f"   Expected: ACK with OpCode 0x{OPCODE_STOP:02X}")
                    print(f"   Received: {response}")
                    print("="*70 + "\n")
                    return False, response
            else:
                print("[ERROR] STOP COMMAND: No response received from device (timeout)")
                print(f"   Device did not respond within {RESPONSE_TIMEOUT} seconds")
                print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            print(f"[ERROR] STOP COMMAND: Error occurred: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print("="*70 + "\n")
            return False, None
    
    def _read_packet(self, timeout: float = RESPONSE_TIMEOUT) -> bytes:
        """
        Read a complete packet (22 bytes) from serial port
        Reads one byte at a time, looks for START_BYTE and END_BYTE
        
        Args:
            timeout: Maximum time to wait for packet (seconds)
            
        Returns:
            bytes: Complete packet (22 bytes)
            
        Raises:
            TimeoutError: If packet not received within timeout
        """
        start_time = time.time()
        buffer = bytearray()
        
        while (time.time() - start_time) < timeout:
            waiting = self.ser.in_waiting
            if waiting:
                # Bulk read, then run the SAME per-byte framing below. Reading one
                # byte per loop and sleeping 0.01 s after each of them capped this
                # at ~65 bytes/s (Windows sleep granularity is ~15 ms) against a
                # device sending ~11,000 bytes/s, so the ACK never arrived and the
                # full timeout was burned on the GUI thread.
                chunk = self.ser.read(waiting)
                if not chunk:
                    continue

                for byte_val in chunk:
                    if not buffer:
                        if byte_val == START_BYTE:
                            buffer.append(byte_val)
                    else:
                        buffer.append(byte_val)
                        if len(buffer) == FRAME_LEN:
                            if buffer[-1] == END_BYTE:
                                return bytes(buffer)
                            buffer.clear()
                # Bytes were consumed this pass: look again before sleeping.
                continue

            # Nothing at the port — yield briefly rather than spinning.
            time.sleep(0.002)
        
        raise TimeoutError("Timeout waiting for packet")
    
    def _wait_for_ack(
        self, expected_opcode: int, timeout: float = 3.0, allow_device_code_0x20: bool = False
    ) -> bytes:
        """
        Wait for ACK frame, filtering out ECG streaming frames (0x20)
        
        Args:
            expected_opcode: Expected opcode in ACK response (byte 5)
            timeout: Maximum time to wait (seconds)
            allow_device_code_0x20: If True, also accept device-specific ACK frames where
                byte[3] == 0x20 and byte[5] == expected_opcode.
            
        Returns:
            bytes: ACK frame
            
        Raises:
            TimeoutError: If ACK not received within timeout
        """
        ECG_STREAM = 0x20
        ACK_CODE = 0x21
        
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                frame = self._read_packet(timeout=0.5)
            except TimeoutError:
                continue
            code = frame[3]

            # Standard ACK
            if code == ACK_CODE and frame[5] == expected_opcode:
                return frame

            # Device-specific ACK (some firmware uses 0x20 for ACK too)
            if allow_device_code_0x20 and code == ECG_STREAM and frame[5] == expected_opcode:
                return frame

            # Otherwise ignore (includes normal streaming frames with code 0x20)
            continue
        
        raise TimeoutError(f"No ACK for opcode 0x{expected_opcode:02X}")
    
    def _wait_for_data(self, timeout: float = 3.0, expected_code: int = DATA_CODE_VERSION) -> bytes:
        """
        Wait for DATA frame, filtering out ECG streaming frames (0x20)
        
        Args:
            timeout: Maximum time to wait (seconds)
            expected_code: Expected DATA code in byte 3 (default: VERSION 0x24)
            
        Returns:
            bytes: DATA frame
            
        Raises:
            TimeoutError: If DATA frame not received within timeout
        """
        ECG_STREAM = 0x20
        DATA_CODE = int(expected_code) & 0xFF
        
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                frame = self._read_packet(timeout=0.5)
            except TimeoutError:
                continue
            code = frame[3]
            
            # Ignore ECG streaming frames
            if code == ECG_STREAM:
                continue
            
            if code == DATA_CODE:
                return frame
        
        raise TimeoutError("No DATA frame received")
    
    def _calc_checksum(self, pkt: bytes) -> int:
        """
        Calculate checksum for packet (sum of bytes 1-20, masked to 8 bits)
        
        Args:
            pkt: Packet bytes (22 bytes)
            
        Returns:
            int: Checksum value (0-255)
        """
        return sum(pkt[1:21]) & 0xFF
    
    def _decode_version_payload(self, payload: bytes) -> str:
        """
        Decode version payload as hex string
        Version data is NOT ASCII, best reliable representation = hex string
        
        Args:
            payload: 16 bytes of version data (bytes 5-20 of the frame)
            
        Returns:
            str: Hex string representation of version (uppercase)
        """
        return payload.hex().upper()
    
    def _parse_packet(self, pkt: bytes) -> Dict:
        """
        Parse packet into dictionary
        
        Args:
            pkt: Packet bytes (22 bytes)
            
        Returns:
            dict: Parsed packet with keys: counter, length, code, checksum, data, raw
        """
        return {
            "counter": pkt[1],
            "length": pkt[2],
            "code": pkt[3],
            "checksum": pkt[4],
            "data": pkt[5:21],
            "raw": pkt
        }
    
    def _build_simple_packet(self, counter: int, opcode: int) -> bytes:
        """
        Build a simple command packet
        
        Args:
            counter: Packet counter
            opcode: Command opcode
            
        Returns:
            bytes: Complete packet (22 bytes)
        """
        pkt = bytearray(22)
        pkt[0] = START_BYTE
        pkt[1] = counter & 0xFF
        pkt[2] = 0x11  # LEN_BYTE
        pkt[3] = opcode
        pkt[4] = 0x00
        pkt[5:21] = bytes(16)
        pkt[21] = END_BYTE
        return bytes(pkt)
    
    def _send_stop(self, timeout: float = 3.0, quiet: bool = False) -> bool:
        """
        Send STOP command to put device in IDLE state

        Args:
            timeout: Timeout in seconds
            quiet: If True, suppress console printing
        
        Returns:
            bool: True if STOP ACK received successfully
        """
        self.ser.reset_input_buffer()
        pkt = self._build_simple_packet(0, OPCODE_STOP)
        if not quiet:
            print("[SEND] STOP ->", pkt.hex(" ").upper())
        self.ser.write(pkt)
        self.ser.flush()
        
        ack = self._wait_for_ack(OPCODE_STOP, timeout=timeout)
        if not quiet:
            print("[RECV] STOP ACK <-", ack.hex(" ").upper())
        return True
    
    def send_version_command(self, counter: int = 0, timeout: float = 1.0, retries: int = 2, quiet: bool = False) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Hardware protocol:
        App -> 0x14
        Device -> 0x21 (ACK, byte[5]=0x14)
        Device -> 0x24 (DATA, bytes[5:21]=ASCII version)
        
        First sends STOP to put device in IDLE, then requests version.
        Filters out ECG streaming frames (0x20) that might interfere.
        
        Args:
            counter: Packet counter (default: 0, but uses 1 for VERSION packet)
            timeout: Timeout in seconds (increased for stability)
            retries: Number of retry attempts
            quiet: If True, suppress console printing
        
        Returns:
            tuple: (success: bool, version_string: str or None, response: dict or None)
        """
        for attempt in range(retries + 1):
            if attempt > 0:
                if not quiet:
                    print(f"[RETRY] Retrying VERSION command (attempt {attempt + 1}/{retries + 1})...")
                time.sleep(0.2) # Wait before retry

            if not quiet:
                print("\n" + "="*60)
                print(f"[SCAN] VERSION COMMAND (Attempt {attempt + 1}): Requesting device version...")
                print("="*60)
            
            CMD_VERSION = OPCODE_VERSION  # 0x13
            
            try:
                # First, stop the device if it's streaming
                try:
                    self._send_stop(timeout=timeout, quiet=quiet)
                    if not quiet:
                        print("[OK] STOP confirmed")
                    time.sleep(0.1)
                except TimeoutError:
                    # Device might already be stopped, continue anyway
                    if not quiet:
                        print("[WARN] STOP ACK timeout (device may already be stopped)")
                
                # Reset buffer before sending VERSION
                self.ser.reset_input_buffer()
                
                # Send VERSION command
                pkt = self._build_simple_packet(1, CMD_VERSION)
                if not quiet:
                    print("[SEND] VERSION ->", pkt.hex(" ").upper())
                self.ser.write(pkt)
                self.ser.flush()
                
                # Wait for ACK (filters out ECG_STREAM frames)
                ack = self._wait_for_ack(CMD_VERSION, timeout=timeout)
                if not quiet:
                    print("[RECV] VERSION ACK <-", ack.hex(" ").upper())
                
                # Wait for DATA (filters out ECG_STREAM frames)
                data = self._wait_for_data(timeout=timeout)
                if not quiet:
                    print("[RECV] VERSION DATA <-", data.hex(" ").upper())
                
                # Decode version from bytes 5-21
                try:
                    version = data[5:21].decode("ascii").rstrip("\x00").strip()
                except UnicodeDecodeError:
                    version = data[5:21].hex().upper()
                
                if not quiet:
                    print("\n" + "="*60)
                    print("[OK] VERSION COMMAND SUCCESS")
                    print("="*60)
                    print("[OK] DEVICE VERSION:", version)
                    print("="*60 + "\n")
                
                # Create response dict for compatibility
                data_response = {
                    "type": "version_data",
                    "counter": data[1],
                    "length": data[2],
                    "code": data[3],
                    "checksum": data[4],
                    "data": data[5:21],
                }
                
                return True, version, data_response
                    
            except TimeoutError as e:
                if not quiet:
                    print(f"[ERROR] VERSION COMMAND: {e}")
                if attempt == retries:
                    if not quiet:
                        print("="*60 + "\n")
                    return False, None, None
            except Exception as e:
                if not quiet:
                    print(f"[ERROR] VERSION COMMAND: Error occurred: {e}")
                if attempt == retries:
                    if not quiet:
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                        print("="*60 + "\n")
                    return False, None, None
        
        return False, None, None

    def send_machine_serial_command(
        self, counter: int = 0, timeout: float = 1.0, retries: int = 2, quiet: bool = False
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Hardware protocol:
        App -> 0x13
        Device -> 0x21 (ACK, byte[5]=0x13)
        Device -> 0x23 (DATA, bytes[5:21]=ASCII serial number)

        First sends STOP to put device in IDLE, then requests machine serial number.
        Filters out ECG streaming frames (0x20) that might interfere.

        Args:
            counter: Packet counter (kept for API symmetry; this path uses 1 for the request frame)
            timeout: Timeout in seconds
            retries: Number of retry attempts
            quiet: If True, suppress all console output (use for background scanning)

        Returns:
            tuple: (success: bool, serial_number: str or None, response: dict or None)
        """
        _ = counter  # counter is fixed in the request packet for stability/compatibility

        for attempt in range(retries + 1):
            if attempt > 0:
                if not quiet:
                    print(
                        f"[RETRY] Retrying MACHINE SERIAL command (attempt {attempt + 1}/{retries + 1})..."
                    )
                time.sleep(0.2)

            if not quiet:
                print("\n" + "=" * 60)
                print(
                    f"[SCAN] MACHINE SERIAL COMMAND (Attempt {attempt + 1}): Requesting machine serial number..."
                )
                print("=" * 60)

            CMD_SERIAL = OPCODE_MACHINE_SERIAL  # 0x13

            try:
                # Stop streaming first so ACK/DATA aren't mixed with 0x20 frames.
                try:
                    self._send_stop(timeout=timeout)
                    if not quiet:
                        print("[OK] STOP confirmed")
                    time.sleep(0.1)
                except TimeoutError:
                    if not quiet:
                        print("[WARN] STOP ACK timeout (device may already be stopped)")

                self.ser.reset_input_buffer()

                pkt = self._build_simple_packet(1, CMD_SERIAL)
                if not quiet:
                    print("[SEND] MACHINE SERIAL ->", pkt.hex(" ").upper())
                self.ser.write(pkt)
                self.ser.flush()

                ack = self._wait_for_ack(CMD_SERIAL, timeout=timeout, allow_device_code_0x20=True)
                if not quiet:
                    print("[RECV] MACHINE SERIAL ACK <-", ack.hex(" ").upper())

                data = self._wait_for_data(timeout=timeout, expected_code=DATA_CODE_MACHINE_SERIAL)
                if not quiet:
                    print("[RECV] MACHINE SERIAL DATA <-", data.hex(" ").upper())

                try:
                    serial_number = data[5:21].decode("ascii").rstrip("\x00").strip()
                except UnicodeDecodeError:
                    serial_number = data[5:21].hex().upper()

                if not quiet:
                    print("\n" + "=" * 60)
                    print("[OK] MACHINE SERIAL COMMAND SUCCESS")
                    print("=" * 60)
                    print("[OK] MACHINE SERIAL NUMBER:", serial_number)
                    print("=" * 60 + "\n")

                data_response = {
                    "type": "machine_serial_data",
                    "counter": data[1],
                    "length": data[2],
                    "code": data[3],
                    "checksum": data[4],
                    "data": data[5:21],
                }

                return True, serial_number, data_response

            except TimeoutError as e:
                if not quiet:
                    print(f"[ERROR] MACHINE SERIAL COMMAND: {e}")
                if attempt == retries:
                    if not quiet:
                        print("=" * 60 + "\n")
                    return False, None, None
            except Exception as e:
                if not quiet:
                    print(f"[ERROR] MACHINE SERIAL COMMAND: Error occurred: {e}")
                if attempt == retries:
                    if not quiet:
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                        print("=" * 60 + "\n")
                    return False, None, None

        return False, None, None
    
    def send_close_command(self) -> Tuple[bool, Optional[Dict]]:
        """
        Send Software close command (OpCode 0x15)
        
        Returns:
            tuple: (success: bool, response: dict or None)
        """
        print("\n" + "="*70)
        print("[CLOSE] CLOSE COMMAND: Requesting device to close connection")
        print("="*70)
        
        try:
            if not self.ser or not getattr(self.ser, "is_open", False):
                print("ℹ️ CLOSE COMMAND: Serial port already closed; skipping hardware close")
                print("="*70 + "\n")
                return True, None

            # Build and send command
            cmd_packet = self._build_command_packet(OPCODE_CLOSE)
            
            # Log what SOFTWARE is SENDING
            print(self._format_packet_details(cmd_packet, "SOFTWARE -> DEVICE", "CLOSE Command"))
            print(f"[SEND] Software sending CLOSE command to device...")
            
            self.ser.write(cmd_packet)
            self.ser.flush()
            
            print(f"[OK] Command packet written to serial port ({len(cmd_packet)} bytes)")
            print(f"[WAIT] Waiting for device response (timeout: {RESPONSE_TIMEOUT}s)...")
            
            # Wait for ACK
            response_packet = self._read_response()
            if response_packet:
                # Log what DEVICE is SENDING BACK
                print(self._format_packet_details(response_packet, "DEVICE -> SOFTWARE", "CLOSE ACK Response"))
                print(f"[RECV] Device responded with packet ({len(response_packet)} bytes)")
                
                response = self._parse_response(response_packet)
                print(f"[PARSED] Parsed response: {response}")
                
                is_ack = False
                if response["type"] == "ack" and response.get("opcode") == OPCODE_CLOSE:
                    is_ack = True
                elif response.get("code") == 0x20:
                    is_ack = True
                    
                if is_ack:
                    print(f"[OK] CLOSE COMMAND: Success! Device acknowledged CLOSE command")
                    print(f"   ACK OpCode: 0x{response.get('opcode', 0):02X} ({self._get_code_name(response.get('opcode', 0))})")
                    print("="*70 + "\n")
                    return True, response
                else:
                    print(f"[WARN] CLOSE COMMAND: Unexpected response type or OpCode mismatch")
                    print(f"   Expected: ACK with OpCode 0x{OPCODE_CLOSE:02X}")
                    print(f"   Received: {response}")
                    print("="*70 + "\n")
                    return False, response
            else:
                print("[ERROR] CLOSE COMMAND: No response received from device (timeout)")
                print(f"   Device did not respond within {RESPONSE_TIMEOUT} seconds")
                print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            print(f"[ERROR] CLOSE COMMAND: Error occurred: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print("="*70 + "\n")
            return False, None
