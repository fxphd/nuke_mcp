print("Starting Nuke MCP...")
try:
    from nuke_mcp_server import main as server_main
    print("Successfully imported server module")

    def main():
        """Entry point for the nuke-mcp package"""
        print("Calling server_main()")
        server_main()
        print("Server main function returned")

    if __name__ == "__main__":
        print("Running main()")
        main()
except Exception as e:
    print(f"Error during import or execution: {str(e)}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
