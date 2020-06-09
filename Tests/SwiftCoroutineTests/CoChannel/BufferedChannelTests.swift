//
//  BufferedChannelTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class BufferedChannelTests: XCTestCase {
    
    func testMeasure() {
        let group = DispatchGroup()
        measure {
            group.enter()
            let channel = CoChannel<Int>(capacity: 1)
            DispatchQueue.global().startCoroutine {
                for value in channel.makeIterator() {
                    let _ = value
                }
                group.leave()
            }
            DispatchQueue.global().startCoroutine {
                for index in (0..<100_000) {
                    try channel.awaitSend(index)
                }
                channel.close()
            }
            group.wait()
        }
    }
    
    func testBufferType() {
        XCTAssertEqual(_BufferedChannel<Int>(capacity: 0).bufferType, .none)
        XCTAssertEqual(_BufferedChannel<Int>(capacity: 1).bufferType, .buffered(capacity: 1))
        XCTAssertEqual(_BufferedChannel<Int>(capacity: .max).bufferType, .unlimited)
    }
    
    func testInsideCoroutine() {
        let exp = expectation(description: "testInsideCoroutine")
        let channel = _BufferedChannel<Int>(capacity: 1)
        var set = Set<Int>()
        DispatchQueue.global().startCoroutine {
            for value in channel.makeIterator() {
                set.insert(value)
            }
            XCTAssertNil(try? channel.awaitReceive())
            XCTAssertEqual(set.count, 100_000)
            exp.fulfill()
        }
        DispatchQueue.global().startCoroutine {
            for index in (0..<100_000) {
                try channel.awaitSend(index)
            }
            XCTAssertTrue(channel.close())
            XCTAssertThrowError(CoChannelError.closed) {
                try channel.awaitSend(-1)
            }
        }
        wait(for: [exp], timeout: 20)
    }
    
    func testOffer() {
        let exp = expectation(description: "testOffer")
        exp.expectedFulfillmentCount = 4
        let channel = _BufferedChannel<Int>(capacity: 1)
        ImmediateScheduler().startCoroutine {
            XCTAssertEqual(try? channel.awaitReceive(), 1)
            exp.fulfill()
        }
        XCTAssertTrue(channel.offer(1))
        XCTAssertTrue(channel.offer(2))
        XCTAssertFalse(channel.offer(3))
        XCTAssertEqual(channel.count, 1)
        channel.whenReceive {
            XCTAssertEqual(try? $0.get(), 2)
            exp.fulfill()
        }
        XCTAssertTrue(channel.offer(4))
        XCTAssertFalse(channel.isEmpty)
        XCTAssertEqual(channel.poll(), 4)
        XCTAssertTrue(channel.isEmpty)
        channel.whenReceive {
            XCTAssertEqual(try? $0.get(), 6)
            exp.fulfill()
        }
        XCTAssertTrue(channel.offer(6))
        XCTAssertTrue(channel.offer(7))
        XCTAssertTrue(channel.close())
        channel.whenReceive {
            XCTAssertEqual(try? $0.get(), 7)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testFuture() {
        let channal = _BufferedChannel<Int>(capacity: 0)
        channal.whenReceive { XCTAssertEqual($0, .success(1)) }
        channal.sendFuture(.init(result: .success(1)))
        channal.sendFuture(.init(result: .success(2)))
        channal.whenReceive { XCTAssertEqual($0, .success(2)) }
    }

    func testCancel() {
        let exp = expectation(description: "testCancel")
        exp.expectedFulfillmentCount = 2
        let channal = _BufferedChannel<Int>(capacity: 0)
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.canceled) { try channal.awaitSend(0) }
            exp.fulfill()
        }
        XCTAssertFalse(channal.isCanceled)
        channal.cancel()
        XCTAssertThrowError(CoChannelError.canceled) { try channal.awaitReceive() }
        XCTAssertThrowError(CoChannelError.canceled) { try channal.awaitSend(1) }
        XCTAssertFalse(channal.offer(1))
        channal.whenReceive { result in
            XCTAssertThrowError(CoChannelError.canceled) { try result.get() }
            exp.fulfill()
        }
        XCTAssertNil(channal.poll())
        XCTAssertTrue(channal.isCanceled)
        wait(for: [exp], timeout: 1)
    }
    
    func testCancel2() {
        let exp = expectation(description: "testCancel2")
        let channal = _BufferedChannel<Int>(capacity: 0)
        channal.whenReceive { result in
            XCTAssertThrowError(CoChannelError.canceled) { try result.get() }
            exp.fulfill()
        }
        channal.cancel()
        wait(for: [exp], timeout: 1)
    }

    func testClose() {
        let exp = expectation(description: "testCancel")
        exp.expectedFulfillmentCount = 2
        let channal = _BufferedChannel<Int>(capacity: 1)
        channal.whenReceive { result in
            XCTAssertThrowError(CoChannelError.closed) { try result.get() }
            exp.fulfill()
        }
        XCTAssertFalse(channal.isClosed)
        XCTAssertTrue(channal.close())
        XCTAssertFalse(channal.close())
        XCTAssertTrue(channal.isClosed)
        channal.whenReceive { result in
            XCTAssertThrowError(CoChannelError.closed) { try result.get() }
            exp.fulfill()
        }
        channal.sendFuture(.init(result: .success(1)))
        wait(for: [exp], timeout: 1)
    }

    func testDeinit() {
        let exp = expectation(description: "testDeinit")
        var channel: _BufferedChannel<Int>! = .init(capacity: .max)
        weak var _weak = channel
        XCTAssertTrue(channel.offer(1))
        let promise = CoPromise<Int>()
        channel.sendFuture(promise)
        channel = nil
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            promise.success(1)
            XCTAssertNil(_weak)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testDeinit2() {
        let exp = expectation(description: "testDeinit2")
        var channel: _BufferedChannel<Int>! = .init(capacity: .max)
        channel.whenReceive { result in
            XCTAssertThrowError(CoChannelError.canceled) { try result.get() }
            exp.fulfill()
        }
        channel = nil
        wait(for: [exp], timeout: 3)
    }

}
