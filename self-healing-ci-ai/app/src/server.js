const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

// Middleware for JSON parsing
app.use(express.json());

// Health check endpoint
app.get('/', (req, res) => {
  res.json({ 
    ok: true, 
    message: 'Self-Healing CI/CD Demo App is running!',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0'
  });
});

// API endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    pid: process.pid
  });
});

// Simple calculator endpoint for testing
app.post('/api/calculate', (req, res) => {
  const { operation, a, b } = req.body;
  
  if (typeof a !== 'number' || typeof b !== 'number') {
    return res.status(400).json({ error: 'Invalid input: a and b must be numbers' });
  }
  
  let result;
  switch (operation) {
    case 'add':
      result = a + b;
      break;
    case 'subtract':
      result = a - b;
      break;
    case 'multiply':
      result = a * b;
      break;
    case 'divide':
      if (b === 0) {
        return res.status(400).json({ error: 'Division by zero' });
      }
      result = a / b;
      break;
    default:
      return res.status(400).json({ error: 'Invalid operation' });
  }
  
  res.json({ result, operation, inputs: { a, b } });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Start server
const server = app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  console.log(`Health check: http://localhost:${port}/`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Process terminated');
  });
});

module.exports = app;
