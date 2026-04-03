def email_callback(email):
    if email is None:
        return None
    if b"ryan@example.com" in email:
        return b"jewik@chapman.edu"
    return email
