import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import healthRouter from './routes/health.js';
import { notFound, errorHandler } from './middlewares/errors.js';

const app = express();

// 1. Security Headers Middleware
app.use(helmet());

// 2. CORS Config
app.use(cors());

// 3. Request Body Parsers
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 4. Request Logging
if (process.env.NODE_ENV === 'development') {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined'));
}

// 5. Mount API Routes
app.get('/', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'Welcome to the VIGILIX AI Surveillance Backend API'
  });
});

app.use('/api/health', healthRouter);

// 6. 404 & Centralized Error Handling Middlewares
app.use(notFound);
app.use(errorHandler);

export default app;
