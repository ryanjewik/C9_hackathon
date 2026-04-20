-- Sliding window rate limiter
-- KEYS[1] - key for team (e.g. ratelimit:team:<id>)
-- ARGV[1] - now in milliseconds
-- ARGV[2] - window size in milliseconds
-- ARGV[3] - max requests allowed in window

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local min = now - window

redis.call('ZADD', key, now, now)
redis.call('ZREMRANGEBYSCORE', key, 0, min)
local count = redis.call('ZCARD', key)
redis.call('PEXPIRE', key, window + 1000)

return count
