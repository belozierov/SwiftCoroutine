//
//  CoroutineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineTests: XCTestCase {

    func testSyncContext() {
        let exp = XCTOrderedExpectation(count: 5)
        let coroutine1 = Coroutine()
        let coroutine2 = Coroutine()
        coroutine1.start {
            exp.fulfill(0)
            coroutine1.suspend()
            exp.fulfill(3)
        }
        exp.fulfill(1)
        coroutine2.start {
            exp.fulfill(2)
            coroutine1.resume()
            exp.fulfill(4)
        }
        wait(for: exp, timeout: 1)
    }
    
    func testRestartWithDispatcher() {
        let exp = expectation(description: "testRestartWithDispatcher")
        let coroutine = Coroutine()
        coroutine.start {
            XCTAssertTrue(Thread.isMainThread)
            coroutine.restart(with: .global)
            XCTAssertFalse(Thread.isMainThread)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testStates() {
        let exp = expectation(description: "testStates")
        exp.expectedFulfillmentCount = 3
        let coroutine = Coroutine()
        coroutine.addHandler { [unowned coroutine] in
            XCTAssertEqual(coroutine.state, $0 ? .prepared : .suspended)
        }
        coroutine.addHandler { _ in exp.fulfill() }
        XCTAssertEqual(coroutine.state, .prepared)
        coroutine.start {
            XCTAssertEqual(coroutine.state, .running)
            coroutine.suspend {
                XCTAssertEqual(coroutine.state, .suspended)
                exp.fulfill()
            }
            XCTAssertEqual(coroutine.state, .running)
        }
        XCTAssertEqual(coroutine.state, .suspended)
        coroutine.resume()
        XCTAssertEqual(coroutine.state, .prepared)
        wait(for: [exp], timeout: 1)
    }
    
    func testHashable() {
        let coroutine1 = Coroutine()
        let coroutine2 = Coroutine()
        XCTAssertNotEqual(coroutine1, coroutine2)
        XCTAssertNotEqual(coroutine1.hashValue, coroutine2.hashValue)
        XCTAssertEqual(coroutine1, coroutine1)
        XCTAssertEqual(coroutine1.hashValue, coroutine1.hashValue)
    }

}
