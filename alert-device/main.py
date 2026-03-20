from listener import AlertListener
from alert import cleanup


def main():
    print("=" * 50)
    print("  HELMET DETECTION -- ALERT DEVICE")
    print("=" * 50)

    listener = AlertListener()

    try:
        listener.run()
    finally:
        listener.close()
        cleanup()
        print("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()