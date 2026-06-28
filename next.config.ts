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
  // oversized-images, vr, wake-lock, interest-cohort) that trigger browser
  // console warnings. interest-cohort is part of the deprecated Privacy
  // Sandbox and is removed to avoid version-specific warnings.
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
            ].join(", "),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
