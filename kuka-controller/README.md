# KUKA Controller Integration

This directory contains the KRL code and integration notes for vision corrections with KUKA controllers, specifically targeting VKRC2 systems. The TCP helper can run on the Raspberry Pi (recommended) or on the Windows controller PC.

## Architecture

```
Raspberry Pi ----TCP----> Windows PC ----KUKAVARPROXY----> KRL Variables
 (correction_helper.py)                    (port 7000)         (SPS.SUB)
                                                                   |
                                                              $BASE update
```

## Files

### KRL Components

- **`config_additions.dat`** - Global variable declarations to add to `$CONFIG.DAT`
- **`sps_additions.src`** - Background processing code for `SPS.SUB`
- **`VisionRun.src`** - Sample main program with vision correction
- **`VisionRun.dat`** - Point data for the sample program

### Helper

- **`raspberry-pi/src/correction_helper.py`** - TCP server that receives corrections and writes KRL variables. Preferred deployment is on the Raspberry Pi.

## Setup Instructions

### 1. KRL Configuration

1. **Add global variables** to `R1/$CONFIG.DAT`:
   ```krl
   ; Copy content from config_additions.dat
   GLOBAL FRAME BASE_REF = {X 0,Y 0,Z 0,A 0,B 0,C 0}
   GLOBAL FRAME G_CORR_RAW = {X 0,Y 0,Z 0,A 0,B 0,C 0}
   ; ... (see full file)
   ```

2. **Modify SPS.SUB** to include continuous correction processing:
   ```krl
   ; Add functions and cyclic code from sps_additions.src
   ; Place the cyclic vision correction code in the main SPS loop
   ```

3. **Install sample program**:
   - Copy `VisionRun.src` to `R1/Program/`
   - Copy `VisionRun.dat` to `R1/Program/`
   - Modify points in `VisionRun.dat` for your workspace

### 2. Helper Setup

Option A (recommended) — run on Raspberry Pi:

1. Ensure KUKAVARPROXY is installed and running on the controller PC (port 7000)
2. On the Pi, run:
   - `python3 raspberry-pi/src/correction_helper.py --port 7001 --kuka-ip <KUKA_PC_IP>`

Option B — run on Windows controller PC:

1. Install Python (3.7+)
2. Ensure KUKAVARPROXY is running on port 7000
3. Copy `raspberry-pi/src/correction_helper.py` and run:
   - `python correction_helper.py --port 7001 --kuka-ip 127.0.0.1`

### 3. Network Configuration

1. **Set static IP** for KRC2 Windows PC (recommended):
   - Example: `192.168.1.50/24`

2. **Configure Raspberry Pi** to send corrections to controller IP:
   ```python
   config = SystemConfig()
   config.controller_ip = "192.168.1.50"
   config.controller_port = 7001
   ```

## Operation

### Startup Sequence

1. **Start KUKAVARPROXY** on Windows PC
2. **Start correction helper**: `python correction_helper.py`
3. **Start vision system** on Raspberry Pi
4. **Load and run** `VisionRun` on KUKA controller

### Runtime Behavior

- **SPS cycle (~12ms)**: Applies filtered corrections to `$BASE`
- **Vision updates (~20-50ms)**: Pi sends new corrections via TCP
- **Low-pass filtering**: Smooths corrections with configurable `G_ALPHA`
- **Safety limits**: Clamps corrections to `G_MAX_MM` and `G_MAX_DEG`

### Monitoring

- **KRL variables** for debugging:
  - `G_CORR_COUNT` - Number of corrections applied
  - `G_LAST_UPDATE` - Timestamp of last correction
  - `G_CORR_FILT` - Current filtered correction values

- **Helper logs** show TCP connection status and correction statistics

## Tuning Parameters

### Filter Response

- **`G_ALPHA = 0.2`** - Filter coefficient (higher = more responsive)
- **Cutoff frequency**: ~`G_ALPHA * SPS_rate / (2π)` ≈ 3Hz at 0.2 and 12ms

### Safety Limits

- **`G_MAX_MM = 5.0`** - Maximum translation correction (mm)
- **`G_MAX_DEG = 0.5`** - Maximum rotation correction (degrees)

### Optional Decay

- **`G_DECAY = 0.02`** - Rate to decay corrections when no updates arrive
- **Uncomment decay code** in SPS if corrections should settle to zero

## Troubleshooting

### Vision System Not Connecting

1. Check network connectivity: `ping 192.168.1.50`
2. Verify Windows firewall allows port 7001
3. Ensure correction helper is running

### KUKAVARPROXY Issues

1. Check KUKAVARPROXY is running on port 7000
2. Verify KRL interpreter is running (not in T1/T2 with program stopped)
3. Check variable names match exactly (case sensitive)

### Corrections Not Applied

1. Verify `G_CORR_VALID` becomes TRUE in KRL
2. Check `G_CORR_COUNT` increments
3. Monitor `$BASE` changes during operation
4. Ensure SPS.SUB modifications are active

### Motion Issues

1. If jerky motion: reduce `G_ALPHA` or increase safety limits
2. If sluggish response: increase `G_ALPHA`
3. If corrections too large: check camera calibration and marker positions

## Advanced Configuration

### Custom Points

Modify `VisionRun.dat` with your actual work points:

```krl
P_WORK[1] = {X 100, Y 200, Z 150, A 0, B 90, C 0, S 2, T 35}
```

### Alternative Base Handling

For multiple coordinate systems, capture different `BASE_REF` values:

```krl
; Switch coordinate systems
BASE_REF = BASE_DATA[2]  ; Use different base
G_CORR_FILT = {X 0,Y 0,Z 0,A 0,B 0,C 0}  ; Reset correction
```

### Performance Optimization

- **Reduce `$ADVANCE`** if motion planning interferes with base updates
- **Increase SPS priority** if timing is critical
- **Use dedicated correction variables** for different operations

## Integration with Existing Programs

To add vision correction to existing programs:

1. **Capture reference** at program start:
   ```krl
   BASE_REF = $BASE
   G_CORR_FILT = {X 0,Y 0,Z 0,A 0,B 0,C 0}
   ```

2. **No other changes required** - SPS handles correction automatically

3. **Optional: disable** correction for certain moves:
   ```krl
   ; Temporarily disable
   G_ALPHA_TEMP = G_ALPHA
   G_ALPHA = 0.0
   ; ... critical moves ...
   G_ALPHA = G_ALPHA_TEMP
   ```
