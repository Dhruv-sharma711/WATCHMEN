import dotenv from 'dotenv';
import app from './app.js';
import connectDB from './config/db.js';

// 1. Load environment variables
dotenv.config();

// 2. Establish connection to Database
connectDB();

const PORT = process.env.PORT || 5000;
const NODE_ENV = process.env.NODE_ENV || 'development';

// 3. Start server listener
const server = app.listen(PORT, () => {
  console.log(`[Server] Running in ${NODE_ENV} mode on port: ${PORT}`);
});

// 4. Handle unexpected system exceptions gracefully
process.on('unhandledRejection', (err) => {
  console.error(`[Fatal] Unhandled Rejection: ${err.message}`);
  if (err.stack) console.error(err.stack);
  
  // Gracefully close active server & shutdown process
  server.close(() => {
    process.exit(1);
  });
});
