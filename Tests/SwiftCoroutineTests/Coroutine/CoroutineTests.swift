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
        let coroutine1 = Coroutine {
            exp.fulfill(0)
            try? Coroutine.suspend()
            exp.fulfill(3)
        }
        try? coroutine1.resume()
        exp.fulfill(1)
        try? Coroutine {
            exp.fulfill(2)
            try? coroutine1.resume()
            exp.fulfill(4)
        }.resume()
        wait(for: exp, timeout: 1)
    }
    
//    func testRestartWithDispatcher() {
//        let exp = expectation(description: "testRestartWithDispatcher")
//        let coroutine = Coroutine()
//        coroutine.start {
//            XCTAssertTrue(Thread.isMainThread)
//            coroutine.restart(with: .global)
//            XCTAssertFalse(Thread.isMainThread)
//            exp.fulfill()
//        }
//        wait(for: [exp], timeout: 1)
//    }
//
    
    func testCurrent() {
        XCTAssertNil(try? Coroutine.current())
        XCTAssertFalse(Coroutine.isInsideCoroutine)
        try? Coroutine {
            XCTAssertNotNil(try? Coroutine.current())
            XCTAssertTrue(Coroutine.isInsideCoroutine)
            XCTAssertTrue(try? Coroutine.current().isCurrent)
        }.resume()
    }
    
//    func testStates() {
//        let exp = expectation(description: "testStates")
//        let coroutine = Coroutine {
//            let coroutine = try? Coroutine.current()
//            XCTAssertEqual(coroutine?.state, .running)
//            try? Coroutine.suspend {
//                XCTAssertEqual(coroutine?.state, .suspended)
//                exp.fulfill()
//            }
//            XCTAssertEqual(coroutine?.state, .running)
//        }
//        XCTAssertEqual(coroutine.state, .prepared)
//        try? coroutine.resume()
//        XCTAssertEqual(coroutine.state, .suspended)
//        try? coroutine.resume()
//        XCTAssertEqual(coroutine.state, .finished)
//        wait(for: [exp], timeout: 1)
//    }
    
    func testHashable() {
        let coroutine1 = Coroutine {}
        let coroutine2 = Coroutine {}
        XCTAssertNotEqual(coroutine1, coroutine2)
        XCTAssertNotEqual(coroutine1.hashValue, coroutine2.hashValue)
        XCTAssertEqual(coroutine1, coroutine1)
        XCTAssertEqual(coroutine1.hashValue, coroutine1.hashValue)
    }
    
    func testStackSize() {
        XCTAssertEqual(Coroutine.StackSize(size: 234).size, 234)
        XCTAssertEqual(Coroutine.StackSize.pages(4).size, 4 * .pageSize)
    }

}
