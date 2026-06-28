import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Prisma must be external so its native engine binary is loaded correctly
  // in standalone/Docker deployments.
  serverExternalPackages: ["@prisma/client"],
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  // Clean, modern Permissions-Policy — only includes currently-supported
  // directives. Removes outdated features (ambient-light-sensor, battery,
  // document-domain, layout-animations, legacy-image-formats,
  // oversized-images, vr, wake-lock) that trigger browser console warnings.
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Permissions-Policy",
            value: [
              "camera=()",
              "microphone=()",
              "geolocation=(self)",
              "interest-cohort=()",
            ].join(", "),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
