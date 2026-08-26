/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost"
      },
      {
        protocol: "http",
        hostname: "127.0.0.1"
      },
      {
        protocol: "https",
        hostname: "api.dicebear.com"
      }
    ]
  },

  // Speed up dev: skip full type-checking during compile (run tsc separately if needed)
  typescript: {
    ignoreBuildErrors: false,
  },

  // Webpack persistent cache (faster cold starts after first run)
  webpack: (config, { dev }) => {
    if (dev) {
      config.cache = {
        type: "filesystem",
      };
    }
    return config;
  },
};

export default nextConfig;
