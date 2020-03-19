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
    
    func testContextSuspendToPoint() {
        let exp = XCTOrderedExpectation(count: 5)
        let context1 = CoroutineContext(stackSize: 8 * .pageSize)
        let context2 = CoroutineContext(stackSize: 8 * .pageSize)
        let savePoint = UnsafeMutablePointer<Int32>.allocate(capacity: .environmentSize)
        context1.block = {
            exp.fulfill(0)
            context1.suspend(to: savePoint)
            exp.fulfill(3)
        }
        context2.block = {
            exp.fulfill(2)
            XCTAssertTrue(context1.resume(from: savePoint))
            exp.fulfill(4)
        }
        XCTAssertFalse(context1.start())
        exp.fulfill(1)
        XCTAssertTrue(context2.start())
        wait(for: exp, timeout: 1)
    }
    
    func testContextSuspendToEnv() {
        let exp = XCTOrderedExpectation(count: 5)
        let context1 = CoroutineContext(stackSize: 8 * .pageSize)
        let context2 = CoroutineContext(stackSize: 8 * .pageSize)
        let env = UnsafeMutablePointer<CoroutineContext.SuspendData>.allocate(capacity: 1)
        env.initialize(to: .init())
        context1.block = {
            exp.fulfill(0)
            context1.suspend(to: env)
            exp.fulfill(3)
        }
        context2.block = {
            exp.fulfill(2)
            XCTAssertTrue(context1.resume(from: env.pointee.env))
            exp.fulfill(4)
        }
        XCTAssertFalse(context1.start())
        exp.fulfill(1)
        XCTAssertTrue(context2.start())
        XCTAssertTrue(env.pointee.sp < context1.stackTop)
        XCTAssertTrue(env.pointee.sp > context1.stackTop - context1.stackSize)
        wait(for: exp, timeout: 1)
    }
    
}


