#!/usr/bin/env node

const http = require('http');

// Test configuration
const TEST_CONFIG = {
  host: 'localhost',
  port: 3000,
  timeout: 5000
};

// Check if we should fail the test (for demo purposes)
const shouldFail = process.env.FAIL_TEST === '1';

console.log('🧪 Starting Self-Healing CI/CD Demo Tests...');
console.log(`📊 FAIL_TEST environment variable: ${process.env.FAIL_TEST || 'not set'}`);

if (shouldFail) {
  console.log('❌ FAIL_TEST=1 detected - simulating test failure');
  console.log('💡 This will trigger the AI analyzer to create a fix');
  process.exit(1);
}

// Track test results
let passedTests = 0;
let totalTests = 0;

function runTest(name, testFn) {
  totalTests++;
  console.log(`\n🔍 Running test: ${name}`);
  
  return new Promise((resolve) => {
    testFn()
      .then(() => {
        passedTests++;
        console.log(`✅ PASS: ${name}`);
        resolve(true);
      })
      .catch((error) => {
        console.log(`❌ FAIL: ${name}`);
        console.log(`   Error: ${error.message}`);
        resolve(false);
      });
  });
}

// Test helper function
function makeRequest(path = '/', method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: TEST_CONFIG.host,
      port: TEST_CONFIG.port,
      path: path,
      method: method,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: TEST_CONFIG.timeout
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(body);
          resolve({ statusCode: res.statusCode, body: parsed });
        } catch (e) {
          resolve({ statusCode: res.statusCode, body: body });
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => reject(new Error('Request timeout')));

    if (data) {
      req.write(JSON.stringify(data));
    }

    req.end();
  });
}

// Start the server for testing
const app = require('../src/server.js');

// Wait for server to start
setTimeout(async () => {
  try {
    // Test 1: Health check endpoint
    await runTest('Health check endpoint', async () => {
      const response = await makeRequest('/');
      if (response.statusCode !== 200) {
        throw new Error(`Expected status 200, got ${response.statusCode}`);
      }
      if (!response.body.ok) {
        throw new Error('Health check should return ok: true');
      }
    });

    // Test 2: API health endpoint
    await runTest('API health endpoint', async () => {
      const response = await makeRequest('/api/health');
      if (response.statusCode !== 200) {
        throw new Error(`Expected status 200, got ${response.statusCode}`);
      }
      if (response.body.status !== 'healthy') {
        throw new Error('API health should return status: healthy');
      }
    });

    // Test 3: Calculator endpoint - addition
    await runTest('Calculator addition', async () => {
      const response = await makeRequest('/api/calculate', 'POST', {
        operation: 'add',
        a: 5,
        b: 3
      });
      if (response.statusCode !== 200) {
        throw new Error(`Expected status 200, got ${response.statusCode}`);
      }
      if (response.body.result !== 8) {
        throw new Error(`Expected result 8, got ${response.body.result}`);
      }
    });

    // Test 4: Calculator endpoint - error handling
    await runTest('Calculator error handling', async () => {
      const response = await makeRequest('/api/calculate', 'POST', {
        operation: 'divide',
        a: 10,
        b: 0
      });
      if (response.statusCode !== 400) {
        throw new Error(`Expected status 400, got ${response.statusCode}`);
      }
      if (!response.body.error.includes('Division by zero')) {
        throw new Error('Should return division by zero error');
      }
    });

    // Test 5: 404 handling
    await runTest('404 error handling', async () => {
      const response = await makeRequest('/nonexistent');
      if (response.statusCode !== 404) {
        throw new Error(`Expected status 404, got ${response.statusCode}`);
      }
    });

    // Print test summary
    console.log('\n📋 Test Summary:');
    console.log(`   Total tests: ${totalTests}`);
    console.log(`   Passed: ${passedTests}`);
    console.log(`   Failed: ${totalTests - passedTests}`);

    if (passedTests === totalTests) {
      console.log('\n🎉 All tests passed!');
      process.exit(0);
    } else {
      console.log('\n💥 Some tests failed!');
      process.exit(1);
    }

  } catch (error) {
    console.error('\n💥 Test runner error:', error.message);
    process.exit(1);
  }
}, 1000); // Give server time to start
