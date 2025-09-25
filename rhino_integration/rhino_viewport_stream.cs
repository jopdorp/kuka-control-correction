/*
Rhino3D TCP Viewport Streaming for KUKA Vision System
Use this C# script in a Grasshopper C# Script component

Inputs:  
- enabled (bool): Enable/disable streaming
- server_host (string): Python server host (e.g., "127.0.0.1") 
- server_port (int): Python server port (e.g., 8080)
- fps (double): Streaming rate in frames per second
- viewport_name (string): Name of viewport to capture (optional)
- quality (int): JPEG quality 1-100 (default: 85)

Outputs:
- status (string): Current status message
- fps_actual (double): Actual achieved FPS 
- connected (bool): Connection status

This provides the highest performance streaming via direct TCP connection.
*/

private void RunScript(bool enabled, string server_host, int server_port, double fps, string viewport_name, int quality, ref object status, ref object fps_actual, ref object connected)
{
    if (!enabled)
    {
        DisconnectTCP();
        status = "Streaming disabled";
        fps_actual = 0.0;
        connected = false;
        return;
    }
    
    // Validate inputs
    if (string.IsNullOrEmpty(server_host) || server_port <= 0)
    {
        status = "Error: Valid server host and port required";
        connected = false;
        return;
    }
    
    if (fps <= 0) fps = 20.0; // Default FPS
    if (quality <= 0 || quality > 100) quality = 85;
    
    try
    {
        // Ensure TCP connection
        if (!EnsureTCPConnection(server_host, server_port))
        {
            status = "Error: Cannot connect to server";
            connected = false;
            fps_actual = 0.0;
            return;
        }
        
        connected = true;
        
        // Get viewport
        Rhino.Display.RhinoView view = GetViewport(viewport_name);
        if (view == null)
        {
            status = "Error: No viewport available";
            return;
        }
        
        // Check timing
        double interval = 1000.0 / fps;
        var now = DateTime.Now;
        
        if (!_lastStreamTime.ContainsKey(view.MainViewport.Id) || 
            (now - _lastStreamTime[view.MainViewport.Id]).TotalMilliseconds >= interval)
        {
            // Capture and stream
            var bitmap = view.CaptureToBitmap(new System.Drawing.Size(1920, 1080), true, true, false);
            
            if (bitmap != null)
            {
                byte[] imageBytes = BitmapToJpegBytes(bitmap, quality);
                bitmap.Dispose();
                
                if (imageBytes != null && SendImageTCP(imageBytes))
                {
                    // Update timing
                    var prev = _lastStreamTime.ContainsKey(view.MainViewport.Id) ? 
                               _lastStreamTime[view.MainViewport.Id] : now;
                    _lastStreamTime[view.MainViewport.Id] = now;
                    
                    double actualFps = 1000.0 / (now - prev).TotalMilliseconds;
                    fps_actual = double.IsInfinity(actualFps) ? 0.0 : actualFps;
                    
                    status = $"TCP Streaming @ {actualFps:F1} FPS ({imageBytes.Length / 1024} KB)";
                }
                else
                {
                    status = "Error: Failed to send via TCP";
                    fps_actual = 0.0;
                }
            }
            else
            {
                status = "Error: Failed to capture viewport";
                fps_actual = 0.0;
            }
        }
        else
        {
            status = $"Connected, waiting... (Target: {fps:F1} FPS)";
        }
    }
    catch (Exception ex)
    {
        status = $"Error: {ex.Message}";
        fps_actual = 0.0;
        connected = false;
        DisconnectTCP();
    }
}

// Class-level variables
private static Dictionary<Guid, DateTime> _lastStreamTime = new Dictionary<Guid, DateTime>();
private static System.Net.Sockets.TcpClient _tcpClient = null;
private static System.Net.Sockets.NetworkStream _networkStream = null;
private static string _currentHost = null;
private static int _currentPort = 0;

private bool EnsureTCPConnection(string host, int port)
{
    // Check if we need to reconnect
    if (_tcpClient == null || !_tcpClient.Connected || 
        _currentHost != host || _currentPort != port)
    {
        DisconnectTCP();
        
        try
        {
            _tcpClient = new System.Net.Sockets.TcpClient();
            _tcpClient.Connect(host, port);
            _networkStream = _tcpClient.GetStream();
            _currentHost = host;
            _currentPort = port;
            return true;
        }
        catch
        {
            DisconnectTCP();
            return false;
        }
    }
    
    return _tcpClient.Connected;
}

private void DisconnectTCP()
{
    try
    {
        if (_networkStream != null)
        {
            _networkStream.Close();
            _networkStream = null;
        }
        
        if (_tcpClient != null)
        {
            _tcpClient.Close();
            _tcpClient = null;
        }
    }
    catch { }
    
    _currentHost = null;
    _currentPort = 0;
}

private bool SendImageTCP(byte[] imageBytes)
{
    try
    {
        if (_networkStream == null || !_networkStream.CanWrite)
            return false;
        
        // Send length header (8 bytes)
        byte[] lengthBytes = BitConverter.GetBytes((long)imageBytes.Length);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(lengthBytes); // Convert to big-endian
        
        _networkStream.Write(lengthBytes, 0, 8);
        
        // Send image data
        _networkStream.Write(imageBytes, 0, imageBytes.Length);
        _networkStream.Flush();
        
        return true;
    }
    catch
    {
        return false;
    }
}

private Rhino.Display.RhinoView GetViewport(string viewport_name)
{
    if (!string.IsNullOrEmpty(viewport_name))
    {
        foreach (var v in Rhino.RhinoDoc.ActiveDoc.Views)
        {
            if (v.MainViewport.Name.Equals(viewport_name, StringComparison.OrdinalIgnoreCase))
                return v;
        }
        return null;
    }
    return Rhino.RhinoDoc.ActiveDoc.Views.ActiveView;
}

private byte[] BitmapToJpegBytes(System.Drawing.Bitmap bitmap, int quality)
{
    try
    {
        using (var stream = new System.IO.MemoryStream())
        {
            var encoderParams = new System.Drawing.Imaging.EncoderParameters(1);
            encoderParams.Param[0] = new System.Drawing.Imaging.EncoderParameter(
                System.Drawing.Imaging.Encoder.Quality, (long)quality);
            
            var jpegEncoder = GetEncoder(System.Drawing.Imaging.ImageFormat.Jpeg);
            bitmap.Save(stream, jpegEncoder, encoderParams);
            return stream.ToArray();
        }
    }
    catch
    {
        return null;
    }
}

private System.Drawing.Imaging.ImageCodecInfo GetEncoder(System.Drawing.Imaging.ImageFormat format)
{
    var codecs = System.Drawing.Imaging.ImageCodecInfo.GetImageDecoders();
    foreach (var codec in codecs)
    {
        if (codec.FormatID == format.Guid)
            return codec;
    }
    return null;
}
