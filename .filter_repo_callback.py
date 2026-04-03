def commit_callback(commit):
    if b"ryan@example.com" in commit.author_email:
        commit.author_email = b"jewik@chapman.edu"
        commit.author_name = b"Ryan Jewik"
    if b"ryan@example.com" in commit.committer_email:
        commit.committer_email = b"jewik@chapman.edu"
        commit.committer_name = b"Ryan Jewik"
