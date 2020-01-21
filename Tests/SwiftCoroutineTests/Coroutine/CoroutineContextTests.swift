//
//  CoroutineContextTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 13.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineContextTests: XCTestCase {
    
    func testInitWithPageGuard() {
        let stackSize = 8 * .pageSize
        let context = CoroutineContext(stackSize: stackSize, guardPage: true)
        XCTAssertEqual(context.stackSize, stackSize)
        XCTAssertTrue(context.haveGuardPage)
    }
    
    func testInitWithoutPageGuard() {
        let stackSize = 8 * .pageSize
        let context = CoroutineContext(stackSize: stackSize, guardPage: false)
        XCTAssertEqual(context.stackSize, stackSize)
        XCTAssertFalse(context.haveGuardPage)
    }
    
    func testContext() {
        let exp = XCTOrderedExpectation(count: 5)
        let context1 = CoroutineContext(stackSize: 8 * .pageSize)
        let context2 = CoroutineContext(stackSize: 8 * .pageSize)
        XCTAssertFalse(context1.start {
            exp.fulfill(0)
            context1.suspend()
            exp.fulfill(3)
        })
        exp.fulfill(1)
        XCTAssertTrue(context2.start {
            exp.fulfill(2)
            XCTAssertTrue(context1.resume())
            exp.fulfill(4)
        })
        wait(for: exp, timeout: 1)
    }
    
}


