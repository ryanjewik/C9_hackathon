def commit_callback(commit):
    try:
        if commit.author_email and b"ryan@example.com" in commit.author_email:
            commit.author_email = b"jewik@chapman.edu"
            commit.author_name = b"Ryan Jewik"
        if commit.committer_email and b"ryan@example.com" in commit.committer_email:
            commit.committer_email = b"jewik@chapman.edu"
            commit.committer_name = b"Ryan Jewik"
    except Exception as e:
        pass
