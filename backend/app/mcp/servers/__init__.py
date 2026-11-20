from app.mcp.servers import analytics_server, jd_server, question_server, resume_server

SERVERS = {
    "resume": resume_server.mcp,
    "questions": question_server.mcp,
    "analytics": analytics_server.mcp,
    "jd": jd_server.mcp,
}
