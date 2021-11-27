import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

Author_name = "bapatlaayyanna"
Repo_name = "Transfer_Learning"
setuptools.setup(
    name="src",
    version="0.0.1",
    author=Author_name,
    author_email=f"{Author_name}@gmail.com",
    description="Transfer Learing Demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{Author_name}/{Repo_name}",
    packages=["src"],
    license="MIT",
    python_requires=">=3.6",
    install_requires=[]
)