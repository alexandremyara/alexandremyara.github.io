rm -rf public resources/_gen
npm ci
HUGO_ENVIRONMENT=production TZ=America/Los_Angeles \
hugo --config hugo.yaml --gc --minify --baseURL "https://alexandremyara.github.io/"
python3 -m http.server 8080 --directory public


