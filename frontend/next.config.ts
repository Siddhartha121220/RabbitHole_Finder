import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // deck.gl ships modern ESM; let Next compile it for the browser bundle.
  transpilePackages: ["@deck.gl/core", "@deck.gl/layers", "@deck.gl/react"],
};

export default nextConfig;
